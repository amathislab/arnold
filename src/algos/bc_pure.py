import torch
import gym
import dataclasses
import tqdm
import numpy as np
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
from torch.utils import data as th_data
from stable_baselines3.common import policies, torch_layers, utils, vec_env
from imitation.algorithms import base as algo_base
from imitation.policies import base as policy_base
from imitation.util import logger as imit_logger
from imitation.data import rollout, types
from imitation.data.types import stack_maybe_dictobs
from imitation.util import util
from imitation.algorithms.bc import (
    BCLogger,
    BC,
    BCTrainingMetrics,
    RolloutStatsComputer,
    BatchIteratorWithEpochEndCallback,
    enumerate_batches
)
from algos.bc_bilateral import make_time_transition_data_loader

@dataclasses.dataclass(frozen=True)
class BCPureTrainingMetrics(BCTrainingMetrics):

    l2_action_loss: torch.Tensor

@dataclasses.dataclass(frozen=True)
class BCPureLossCalculator :

    ent_weight: float
    l2_weight: float
    action_loss_mode: str = "l2"   # "l2" or "neg_log_p"

    def __call__(
        self,
        policy: policies.ActorCriticPolicy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, torch.Tensor],
        ],
        acts: Union[torch.Tensor, np.ndarray],
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
        tensor_obs = types.map_maybe_dict(
            util.safe_to_tensor,
            types.maybe_unwrap_dictobs(obs),
        )
        acts = util.safe_to_tensor(acts)

        # policy.evaluate_actions's type signatures are incorrect.
        # See https://github.com/DLR-RM/stable-baselines3/issues/1679
        # (_, log_prob, entropy) = policy.evaluate_actions(
        #     tensor_obs,  # type: ignore[arg-type]
        #     acts,
        # )
        # prob_true_act = torch.exp(log_prob).mean()
        # log_prob = log_prob.mean()
        # neglogp = -log_prob

        if self.action_loss_mode == "neg_log_p":
            (_, log_prob, entropy) = policy.evaluate_actions(
                tensor_obs,  # type: ignore[arg-type]
                acts,
            )
            prob_true_act = torch.exp(log_prob).mean()
            log_prob = log_prob.mean()
            neglogp = -log_prob
            l2_action_loss = 0
        elif self.action_loss_mode == "l2":
            dist = policy.get_distribution(tensor_obs)
            log_prob = dist.log_prob(acts)
            prob_true_act = torch.exp(log_prob).mean()
            entropy = None
            neglogp = 0
            l2_action_loss = torch.mean((dist.distribution.loc - acts)**2)
        else :
            raise ValueError(f"Invalid action_loss_mode: {self.action_loss_mode}")


        entropy = entropy.mean() if entropy is not None else None

        l2_norms = [torch.sum(torch.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, torch.Tensor)

        ent_loss = -self.ent_weight * (entropy if entropy is not None else torch.zeros(1, device=pred_value.device))
        l2_loss = self.l2_weight * l2_norm
        # loss = neglogp + ent_loss + l2_loss + value_loss
        loss = l2_action_loss + neglogp + ent_loss + l2_loss

        return BCPureTrainingMetrics(
            neglogp=neglogp,
            l2_action_loss=l2_action_loss,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            loss=loss
        )

class BCPure(BC) :

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
        value_weight: float = 0.0,
        device: Union[str, torch.device] = "auto",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        **unused_kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            rng=rng,
            policy=policy,
            demonstrations=demonstrations,
            batch_size=batch_size,
            minibatch_size=minibatch_size,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            ent_weight=ent_weight,
            l2_weight=l2_weight,
            device=device,
            custom_logger=custom_logger,
        )
        self.loss_calculator = BCPureLossCalculator(ent_weight, l2_weight, action_loss_mode)
