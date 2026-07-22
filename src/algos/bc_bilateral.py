import numpy as np
import torch
import gym
import tqdm
import dataclasses
from imitation.algorithms import base as algo_base
from imitation.data.types import stack_maybe_dictobs, DictObs
from imitation.policies import base as policy_base
from imitation.util import logger as imit_logger
from imitation.util import util
from stable_baselines3.common import policies, torch_layers, utils, vec_env
from torch.utils import data as th_data
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    Sequence,
    cast
)
from imitation.algorithms.bc import (
    BCLogger,
    BC,
    BCTrainingMetrics,
    RolloutStatsComputer,
    BatchIteratorWithEpochEndCallback,
    enumerate_batches
)
from imitation.data import rollout, types
from models.ppo.policies import MuscleTransformerPolicy, BilateralMuscleTransformerPolicy
from torch_utils import ContinuousBatchSampler, generate_token_mask_vectorized, select_by_mask

@dataclasses.dataclass(frozen=True)
class BCBilateralTrainingMetrics(BCTrainingMetrics):

    obs_prediction_loss: torch.Tensor

@dataclasses.dataclass(frozen=True)
class BCBilateralLossCalculator:
    """Functor to compute the loss used in Behavior Cloning."""

    ent_weight: float
    l2_weight: float
    obs_pred_weight: float
    action_loss_mode: str = "l2"
    masking_ratio: float = 0.5

    def __call__(
        self,
        policy: BilateralMuscleTransformerPolicy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, torch.Tensor],
        ],
        next_obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, torch.Tensor],
        ],
        dones: Union[types.AnyTensor, np.ndarray],
        acts: Union[types.AnyTensor, np.ndarray],
    ) -> BCTrainingMetrics:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy whose loss is being computed.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            A BCTrainingMetrics object with the loss and all the components it
            consists of.
        """
        if self.action_loss_mode == "l2" :
            pass
            # raise NotImplementedError("L2 loss for actions is not implemented yet.")
        elif self.action_loss_mode == "neg_log_p" :
            pass
        else :
            raise ValueError(f"Invalid action loss mode: {self.action_loss_mode}")

        tensor_obs = types.map_maybe_dict(
            util.safe_to_tensor,
            types.maybe_unwrap_dictobs(obs),
        )
        dones = util.safe_to_tensor(dones)
        acts = util.safe_to_tensor(acts)

        obs_mask = generate_token_mask_vectorized(1, tensor_obs["obs"].shape[0], int(self.masking_ratio * tensor_obs["obs"].shape[0])).reshape(-1)
        act_mask = obs_mask.clone()
        # obs_mask = torch.zeros((tensor_obs["obs"].shape[0],), device=tensor_obs["obs"].device, dtype=torch.bool)
        # obs_mask[:int(self.masking_ratio * obs_mask.shape[0])+1] = 1
        # obs_mask = obs_mask[torch.randperm(obs_mask.shape[0])].contiguous()
        # obs_mask[-1] = 1    # always predict the last observation
        # act_mask = torch.zeros((acts.shape[0],), device=acts.device, dtype=torch.bool)
        # act_mask[:int(self.masking_ratio * act_mask.shape[0])+1] = 1
        # act_mask = act_mask[torch.randperm(act_mask.shape[0])].contiguous()

        # policy.evaluate_actions's type signatures are incorrect.
        # See https://github.com/DLR-RM/stable-baselines3/issues/1679

        if self.action_loss_mode == "neg_log_p":
            _, log_prob, entropy = policy.evaluate_actions_bilateral(
                tensor_obs,
                dones,
                acts,
                obs_mask = obs_mask,
                act_mask = act_mask
                # unsqueeze_dict_tensor(tensor_obs, 0),
                # dones.unsqueeze(0),
                # acts.unsqueeze(0)
            )
            prob_true_act = torch.exp(log_prob).mean()
            log_prob = log_prob.mean()
            neglogp = -log_prob
            l2_action_loss = 0
        elif self.action_loss_mode == "l2":
            acts_maksed = select_by_mask(acts, act_mask)
            dist = policy.get_distribution_bilateral(
                tensor_obs,
                dones,
                obs_mask = obs_mask,
                act_mask = act_mask
            )
            log_prob = dist.log_prob(acts_maksed)
            prob_true_act = torch.exp(log_prob).mean()
            entropy = None
            neglogp = 0
            l2_action_loss = torch.mean((dist.distribution.loc - acts_maksed)**2)

        # (_, log_prob, entropy) = policy.evaluate_actions(
        #     tensor_obs,  # type: ignore[arg-type]
        #     acts,
        # )
        # next_obs_pred = policy.predict_next_obs(
        #     tensor_obs,
        #     acts
        # )
        prob_true_act = torch.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [torch.sum(torch.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)

        ent_loss = -self.ent_weight * (entropy if entropy is not None else torch.zeros(1, device=acts.device))
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        # next_obs_pred_loss = torch.nn.functional.mse_loss(next_obs_pred, next_obs["obs"]) * self.obs_pred_weight
        loss = l2_action_loss + neglogp + ent_loss + l2_loss#  + next_obs_pred_loss

        return BCBilateralTrainingMetrics(
            neglogp=neglogp,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            obs_prediction_loss=0,#next_obs_pred_loss,
            loss=loss,
        )

def predict_transitions_collate_fn(
    batch: Sequence[Mapping[str, np.ndarray]],
) -> Mapping[str, Union[np.ndarray, torch.Tensor]]:
    """Custom `torch.utils.data.DataLoader` collate_fn for `TransitionsMinimal`.

    Use this as the `collate_fn` argument to `DataLoader` if using an instance of
    `TransitionsMinimal` as the `dataset` argument.

    Args:
        batch: The batch to collate.

    Returns:
        A collated batch. Uses Torch's default collate function for everything
        except the "infos" key. For "infos", we join all the info dicts into a
        list of dicts. (The default behavior would recursively collate every
        info dict into a single dict, which is incorrect.)
    """
    batch_acts_and_dones = [
        {k: np.array(v) for k, v in sample.items() if k in ["acts", "dones", "rews"]}
        for sample in batch
    ]

    result = th_data.dataloader.default_collate(batch_acts_and_dones)
    assert isinstance(result, dict)
    result["infos"] = [sample["infos"] for sample in batch]
    result["obs"] = stack_maybe_dictobs([sample["obs"] for sample in batch])
    result["next_obs"] = stack_maybe_dictobs([sample["next_obs"] for sample in batch])
    return result

def make_time_transition_data_loader(
    transitions: algo_base.AnyTransitions,
    batch_size: int,
    time_skip: int,
    data_loader_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[types.TransitionMapping]:
    """Converts demonstration data to Torch data loader.

    Args:
        transitions: Transitions expressed directly as a `types.TransitionsMinimal`
            object, a sequence of trajectories, or an iterable of transition
            batches (mappings from keywords to arrays containing observations, etc).
        batch_size: The size of the batch to create. Does not change the batch size
            if `transitions` is already an iterable of transition batches.
        data_loader_kwargs: Arguments to pass to `th_data.DataLoader`.

    Returns:
        An iterable of transition batches.

    Raises:
        ValueError: if `transitions` is an iterable over transition batches with batch
            size not equal to `batch_size`; or if `transitions` is transitions or a
            sequence of trajectories with total timesteps less than `batch_size`.
        TypeError: if `transitions` is an unsupported type.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be positive.")

    kwargs: Mapping[str, Any] = {
        **(data_loader_kwargs or {}),
    }

    def fill_skip(arr: np.ndarray, skip) :

        assert(len(arr.shape) == 1)
        ret = np.zeros_like(arr)
        for i in range(1, np.minimum(skip+1, arr.shape[0])) :
            ret[i:] = np.maximum(ret[i:], arr[:-i])

        return ret

    obs_list = [trans.obs for trans in transitions]
    acts_list = [trans.acts for trans in transitions]
    infos_list = [trans.infos for trans in transitions]
    next_obs_list = [trans.next_obs for trans in transitions]
    dones_list = [trans.dones for trans in transitions]
    rews_list = [trans.rews for trans in transitions]

    cat_transitions = types.TransitionsWithRew(
        obs=DictObs.concatenate(obs_list),
        acts=np.concatenate(acts_list),
        infos=np.concatenate(infos_list),
        next_obs=DictObs.concatenate(next_obs_list),
        dones=fill_skip(np.concatenate(dones_list), time_skip),
        rews=np.concatenate(rews_list)
    )

    return th_data.DataLoader(
        cat_transitions,
        batch_sampler=ContinuousBatchSampler(cat_transitions, batch_size, time_skip),
        collate_fn=predict_transitions_collate_fn,
        **kwargs,
    )

class BCBilateral(algo_base.DemonstrationAlgorithm) :

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        policy: Optional[policies.ActorCriticPolicy] = None,
        demonstrations: Optional[algo_base.AnyTransitions] = None,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        action_loss_mode: str = "l2",
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        obs_pred_weight: float = 1.0,
        device: Union[str, torch.device] = "auto",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        time_skip: int = 1,
        mask_rate: float = 0.5,
        **unused_kwargs: Any,
    ):
        """Builds BC.

        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.
            rng: the random state to use for the random number generator.
            policy: a Stable Baselines3 policy; if unspecified,
                defaults to `FeedForward32Policy`.
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            batch_size: The number of samples in each batch of expert data.
            minibatch_size: size of minibatch to calculate gradients over.
                The gradients are accumulated until `batch_size` examples
                are processed before making an optimization step. This
                is useful in GPU training to reduce memory usage, since
                fewer examples are loaded into memory at once,
                facilitating training with larger batch sizes, but is
                generally slower. Must be a factor of `batch_size`.
                Optional, defaults to `batch_size`.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments, excluding learning rate and
                weight decay, for optimiser construction.
            ent_weight: scaling applied to the policy's entropy regularization.
            l2_weight: scaling applied to the policy's L2 regularization.
            device: name/identity of device to place policy on.
            custom_logger: Where to log to; if None (default), creates a new logger.

        Raises:
            ValueError: If `weight_decay` is specified in `optimizer_kwargs` (use the
                parameter `l2_weight` instead), or if the batch size is not a multiple
                of the minibatch size.
        """
        self._demo_data_loader: Optional[Iterable[types.TransitionMapping]] = None
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
        )
        self._bc_logger = BCLogger(self.logger)

        self.action_space = action_space
        self.observation_space = observation_space

        self.rng = rng

        if policy is None:
            extractor = (
                torch_layers.CombinedExtractor
                if isinstance(observation_space, gym.spaces.Dict)
                else torch_layers.FlattenExtractor
            )
            policy = policy_base.FeedForward32Policy(
                observation_space=observation_space,
                action_space=action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: torch.finfo(torch.float32).max,
                features_extractor_class=extractor,
            )
        self._policy = policy.to(utils.get_device(device))
        # TODO(adam): make policy mandatory and delete observation/action space params?
        assert self.policy.observation_space == self.observation_space
        assert self.policy.action_space == self.action_space

        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )

        self.loss_calculator = BCBilateralLossCalculator(
            ent_weight,
            l2_weight,
            obs_pred_weight,
            action_loss_mode,
            masking_ratio = mask_rate
        )

        self.time_skip = time_skip
        self.mask_rate = mask_rate

    @property
    def policy(self) -> policies.ActorCriticPolicy:
        return self._policy

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        self._demo_data_loader = make_time_transition_data_loader(
            demonstrations,
            self.minibatch_size,
            self.time_skip
        )

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`. Note, that when you specify
        `n_batches` smaller than the number of batches in an epoch, the `on_epoch_end`
        callback will never be called.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            on_epoch_end: Optional callback with no parameters to run at the end of each
                epoch.
            on_batch_end: Optional callback with no parameters to run at the end of each
                batch.
            log_interval: Log stats after every log_interval batches.
            log_rollouts_venv: If not None, then this VecEnv (whose observation and
                actions spaces must match `self.observation_space` and
                `self.action_space`) is used to generate rollout stats, including
                average return and average episode length. If None, then no rollouts
                are generated.
            log_rollouts_n_episodes: Number of rollouts to generate when calculating
                rollout stats. Non-positive number disables rollouts.
            progress_bar: If True, then show a progress bar during training.
            reset_tensorboard: If True, then start plotting to Tensorboard from x=0
                even if `.train()` logged to Tensorboard previously. Has no practical
                effect if `.train()` is being called for the first time.
        """
        if reset_tensorboard:
            self._bc_logger.reset_tensorboard_steps()
        self._bc_logger.log_epoch(0)

        compute_rollout_stats = RolloutStatsComputer(
            log_rollouts_venv,
            log_rollouts_n_episodes,
        )

        def _on_epoch_end(epoch_number: int):
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(
                    f"Epoch {epoch_number} {total_num_epochs_str}",
                    pos=1,
                )
            self._bc_logger.log_epoch(epoch_number + 1)
            if on_epoch_end is not None:
                on_epoch_end()

        mini_per_batch = self.batch_size // self.minibatch_size
        n_minibatches = n_batches * mini_per_batch if n_batches is not None else None

        assert self._demo_data_loader is not None
        demonstration_batches = BatchIteratorWithEpochEndCallback(
            self._demo_data_loader,
            n_epochs,
            n_minibatches,
            _on_epoch_end,
        )
        batches_with_stats = enumerate_batches(demonstration_batches)
        tqdm_progress_bar: Optional[tqdm.tqdm] = None

        if progress_bar:
            batches_with_stats = tqdm.tqdm(
                batches_with_stats,
                unit="batch",
                total=n_minibatches,
            )
            tqdm_progress_bar = batches_with_stats

        def process_batch():
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_num % log_interval == 0:
                rollout_stats = compute_rollout_stats(self.policy, self.rng)

                self._bc_logger.log_batch(
                    batch_num,
                    minibatch_size,
                    num_samples_so_far,
                    training_metrics,
                    rollout_stats,
                )

            if on_batch_end is not None:
                on_batch_end()

        self.optimizer.zero_grad()
        for (
            batch_num,
            minibatch_size,
            num_samples_so_far,
        ), batch in batches_with_stats:
            obs_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]
            # unwraps the observation if it's a dictobs and converts arrays to tensors
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["obs"]),
            )
            next_obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["next_obs"]),
            )
            # value = util.safe_to_tensor(batch["rews"].float(), device=self.policy.device)
            dones = util.safe_to_tensor(batch["dones"], device=self.policy.device)
            acts = util.safe_to_tensor(batch["acts"], device=self.policy.device)
            training_metrics = self.loss_calculator(self.policy, obs_tensor, next_obs_tensor, dones, acts)

            # Renormalise the loss to be averaged over the whole
            # batch size instead of the minibatch size.
            # If there is an incomplete batch, its gradients will be
            # smaller, which may be helpful for stability.
            loss = training_metrics.loss * minibatch_size / self.batch_size
            loss.backward()

            batch_num = batch_num * self.minibatch_size // self.batch_size
            if num_samples_so_far % self.batch_size == 0:
                process_batch()
        if num_samples_so_far % self.batch_size != 0:
            # if there remains an incomplete batch
            batch_num += 1
            process_batch()
