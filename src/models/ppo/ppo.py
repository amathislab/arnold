import pathlib
import io
from copy import deepcopy
import torch
import warnings
import numpy as np
from gymnasium import spaces
from torch.nn import functional as F
from typing import Dict, Any, Union, Optional, Type, List, Generator, NamedTuple
from stable_baselines3.ppo import PPO
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from stable_baselines3.common.type_aliases import GymEnv, DictRolloutBufferSamples
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.utils import get_system_info, explained_variance, obs_as_tensor, get_schedule_fn
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr
from models.pcgrad_utils import get_param_grad, proj_grad, param_sum, set_param_grad
from algos.dagger_value import policy_list_to_callable
from vocabulary import VOCABULARY
TensorDict = Dict[str, torch.Tensor]

class NestedLSTMDictRolloutBufferSamples(NamedTuple) :

    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    lstm_states: List[Any]
    episode_starts: List[bool]
    obs_dicts: List[Dict[str, Any]]
    env_ids: List[int]

class NestedLSTMDictBuffer(DictRolloutBuffer) :

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )
        self.lstm_states = []
        self.obs_dicts = []
    
    def add(  # type: ignore[override]
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        lstm_states: List[RNNStates],
        obs_dicts: List[Dict[str, Any]],
    ) -> None:

        super().add(
            obs=obs,
            action=action,
            reward=reward,
            episode_start=episode_start,
            value=value,
            log_prob=log_prob,
        )

        self.lstm_states.append(lstm_states)
        self.obs_dicts.append(obs_dicts)
    
    def reset(self) -> None:
        super().reset()
        self.lstm_states = []
        self.obs_dicts = []
    
    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> NestedLSTMDictRolloutBufferSamples:

        gathered_lstm_states = [
            self.lstm_states[i // self.n_envs][i % self.n_envs] for i in batch_inds
        ]
        gathered_episode_starts = [
            self.episode_starts[i // self.n_envs][i % self.n_envs] for i in batch_inds
        ]
        gathered_obs_dicts = [
            self.obs_dicts[i // self.n_envs][i % self.n_envs] for i in batch_inds
        ]

        env_ids = [i % self.n_envs for i in batch_inds]

        return NestedLSTMDictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            lstm_states=gathered_lstm_states,
            episode_starts=gathered_episode_starts,
            obs_dicts=gathered_obs_dicts,
            env_ids=env_ids
        )

def split_lstm_states(lstm_states, envs_per_policy: int = 1) -> List[List[RNNStates]]:

    ret = []
    for lstm_state in lstm_states:
        if lstm_state is None:
            ret += [None for _ in range(envs_per_policy)]
        else :
            for i in range(envs_per_policy):
                ret.append(
                    RNNStates(
                        pi = (
                            lstm_state.pi[0][:, i:i+1, :].contiguous(),
                            lstm_state.pi[1][:, i:i+1, :].contiguous()
                        ),
                        vf = (
                            lstm_state.vf[0][:, i:i+1, :].contiguous(),
                            lstm_state.vf[1][:, i:i+1, :].contiguous()
                        )
                    )
                )

    return ret

class SeparatedValueFinetunePPO(PPO) :

    def _setup_model(self) -> None:

        self.rollout_buffer_class = NestedLSTMDictBuffer

        super()._setup_model()

        self.expert_policy_list = self.policy_kwargs["expert_policy_list"]
        self.train_env_list = self.policy_kwargs["train_env_list"]
        self.vecnormalize_list = self.policy_kwargs["vecnormalize_list"]

        updated_train_env_list = []

        for train_env in self.train_env_list:
            if isinstance(train_env.observation_space, spaces.Dict):
                updated_train_env_list.append(None)
            else:
                # Not an Arnold environment, the obs needs to be converted
                updated_train_env_list.append(train_env)
        self.train_env_list = updated_train_env_list

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: NestedLSTMDictBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        get_expert_action_value_state = policy_list_to_callable(
            self.expert_policy_list,
            self.env,
            self.train_env_list,
            self.vecnormalize_list,
            train_mode=True
        )

        callback.on_rollout_start()

        expert_states = None
        self._last_expert_states = [None for _ in range(env.num_envs)]
        self._last_obs_dict = self.env.get_attr("obs_dict")

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            # env.envs[0].sim.renderer.render_to_window()

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                _, _, log_probs = self.policy(obs_tensor)
                actions, values, expert_states = get_expert_action_value_state(
                    self._last_obs,
                    self._last_obs_dict,
                    self._last_expert_states,
                    self._last_episode_starts,
                )

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            current_obs_dict = self.env.get_attr("obs_dict")

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):

                    terminal_obs = infos[idx]["terminal_observation"]
                    # terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    # import ipdb
                    # ipdb.set_trace()
                    with torch.no_grad():
                        num_envs_per_expert = self.env.num_envs // len(self.expert_policy_list)
                        lst_state_this_env = expert_states[idx // num_envs_per_expert]
                        if lst_state_this_env is not None :
                            terminal_pi_state = (
                                lst_state_this_env.pi[0][:, idx%num_envs_per_expert : idx%num_envs_per_expert + 1, :].contiguous(),
                                lst_state_this_env.pi[1][:, idx%num_envs_per_expert : idx%num_envs_per_expert + 1, :].contiguous(),
                            )
                            terminal_vf_state = (
                                lst_state_this_env.vf[0][:, idx%num_envs_per_expert : idx%num_envs_per_expert + 1, :].contiguous(),
                                lst_state_this_env.vf[1][:, idx%num_envs_per_expert : idx%num_envs_per_expert + 1, :].contiguous(),
                            )
                            terminal_state = RNNStates(pi=terminal_pi_state, vf=terminal_vf_state)
                        else :
                            terminal_state = None
                        
                        terminal_value = get_expert_action_value_state.predict_single_value(
                            terminal_obs,
                            current_obs_dict[idx],
                            idx,
                            terminal_state,
                            False
                        )
                        # terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
                split_lstm_states(deepcopy(self._last_expert_states), get_expert_action_value_state.envs_per_policy),
                deepcopy(self._last_obs_dict)
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            self._last_expert_states = expert_states
            self._last_obs_dict = current_obs_dict

        with torch.no_grad():
            # Compute value for the last timestep
            values = get_expert_action_value_state.predict_values(
                new_obs,
                self.env.get_attr("obs_dict"),
                expert_states,
                self._last_episode_starts,
            )  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True
    
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        get_expert_action_value_state = policy_list_to_callable(
            self.expert_policy_list,
            self.env,
            self.train_env_list,
            self.vecnormalize_list,
            train_mode=True
        )

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                observations = rollout_data.observations
                actions = rollout_data.actions
                env_ids = rollout_data.env_ids
                obs_dicts = rollout_data.obs_dicts
                episode_starts = rollout_data.episode_starts
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                lstm_states = rollout_data.lstm_states
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = get_expert_action_value_state.evaluate_actions(
                    env_ids,
                    observations,
                    actions,
                    obs_dicts,
                    lstm_states,
                    episode_starts
                )
                # values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                # values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred.flatten())
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

class MultiEnvPPO(PPO):
    @classmethod
    def load(
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        """Same as load from standard PPO, but it accepts a new
        environment with different observation and action spaces.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        if print_system_info:
            print("== CURRENT SYSTEM INFO ==")
            get_system_info()

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]
            # backward compatibility, convert to new format
            if (
                "net_arch" in data["policy_kwargs"]
                and len(data["policy_kwargs"]["net_arch"]) > 0
            ):
                saved_net_arch = data["policy_kwargs"]["net_arch"]
                if isinstance(saved_net_arch, list) and isinstance(
                    saved_net_arch[0], dict
                ):
                    data["policy_kwargs"]["net_arch"] = saved_net_arch[0]

        if (
            "policy_kwargs" in kwargs
            and kwargs["policy_kwargs"] != data["policy_kwargs"]
        ):
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if env is not None:
            # Wrap first if needed
            env = cls._wrap_env(env, data["verbose"])
            # Set the observation and action spaces to those of the loaded environment
            data["observation_space"] = _convert_space(env.observation_space)
            data["action_space"] = _convert_space(env.action_space)
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            if "observation_space" not in data or "action_space" not in data:
                raise KeyError(
                    "The observation_space and action_space were not given, can't verify new environments"
                )

            # Gym -> Gymnasium space conversion
            for key in {"observation_space", "action_space"}:
                data[key] = _convert_space(data[key])

            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            policy=data["policy_class"],
            env=env,
            device=device,
            _init_setup_model=False,  # type: ignore[call-arg]
        )

        # load parameters
        if "policy_kwargs" in data and "features_extractor_kwargs" in data["policy_kwargs"]:
            data["policy_kwargs"]["features_extractor_kwargs"]["vocabulary"] = custom_objects.get("vocabulary")
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()
        model.policy.to(device)

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
            if hasattr(model.policy, "set_vocabulary"):
                model.policy.set_vocabulary(VOCABULARY)
        except RuntimeError as e:
            # Patch to load Policy saved using SB3 < 1.7.0
            # the error is probably due to old policy being loaded
            # See https://github.com/DLR-RM/stable-baselines3/issues/1233
            if "pi_features_extractor" in str(
                e
            ) and "Missing key(s) in state_dict" in str(e):
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "You are probably loading a model saved with SB3 < 1.7.0, "
                    "we deactivated exact_match so you can save the model "
                    "again to avoid issues in the future "
                    "(see https://github.com/DLR-RM/stable-baselines3/issues/1233 for more info). "
                    f"Original error: {e} \n"
                    "Note: the model should still work fine, this only a warning."
                )
            else:
                model.set_parameters(params, exact_match=False, device=device)
                warnings.warn(
                    "Some state dict are not matching with the current model"
                )
        # put other pytorch variables back in place
        if pytorch_variables is not None:
            for name in pytorch_variables:
                # Skip if PyTorch variable was not defined (to ensure backward compatibility).
                # This happens when using SAC/TQC.
                # SAC has an entropy coefficient which can be fixed or optimized.
                # If it is optimized, an additional PyTorch variable `log_ent_coef` is defined,
                # otherwise it is initialized to `None`.
                if pytorch_variables[name] is None:
                    continue
                # Set the data attribute directly to avoid issue when using optimizers
                # See https://github.com/DLR-RM/stable-baselines3/issues/391
                recursive_setattr(model, f"{name}.data", pytorch_variables[name].data)

        # Sample gSDE exploration matrix, so it uses the right device
        # see issue #44
        if model.use_sde:
            model.policy.reset_noise()  # type: ignore[operator]
        return model

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        value_real_mean_list, value_real_std_list = [], []
        value_pred_mean_list, value_pred_std_list = [], []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                value_pred_mean = torch.mean(values_pred.detach().cpu()).item()
                value_pred_std = torch.std(values_pred.detach().cpu()).item()
                value_real_mean = torch.mean(rollout_data.returns.detach().cpu()).item()
                value_real_std = torch.std(rollout_data.returns.detach().cpu()).item()
                value_real_mean_list.append(value_real_mean)
                value_real_std_list.append(value_real_std)
                value_pred_mean_list.append(value_pred_mean)
                value_pred_std_list.append(value_pred_std)


                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/value_real_mean", np.mean(value_real_mean_list))
        self.logger.record("train/value_real_std", np.mean(value_real_std_list))
        self.logger.record("train/value_pred_mean", np.mean(value_pred_mean_list))
        self.logger.record("train/value_pred_std", np.mean(value_pred_std_list))
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


# def train(self) -> None:
#     """
#     Update policy using the currently gathered rollout buffer.
#     """
#     # Switch to train mode (this affects batch norm / dropout)
#     self.policy.set_training_mode(True)
#     # Update optimizer learning rate
#     self._update_learning_rate(self.policy.optimizer)
#     # Compute current clip range
#     clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
#     # Optional: clip range for the value function
#     if self.clip_range_vf is not None:
#         clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

#     entropy_losses = []
#     pg_losses, value_losses = [], []
#     clip_fractions = []

#     continue_training = True
#     # train for n_epochs epochs
#     for epoch in range(self.n_epochs):
#         approx_kl_divs = []
#         # Do a complete pass on the rollout buffer
#         for rollout_data in self.rollout_buffer.get(self.batch_size):
#             actions = rollout_data.actions
#             if isinstance(self.action_space, spaces.Discrete):
#                 # Convert discrete action from float to long
#                 actions = rollout_data.actions.long().flatten()

#             # Re-sample the noise matrix because the log_std has changed
#             if self.use_sde:
#                 self.policy.reset_noise(self.batch_size)

#             values, log_prob, entropy = self.policy.evaluate_actions(
#                 rollout_data.observations, actions
#             )
#             values = values.flatten()
#             # Normalize advantage
#             advantages = rollout_data.advantages
#             # Normalization does not make sense if mini batchsize == 1, see GH issue #325
#             if self.normalize_advantage and len(advantages) > 1:
#                 advantages = (advantages - advantages.mean()) / (
#                     advantages.std() + 1e-8
#                 )

#             # ratio between old and new policy, should be one at the first iteration
#             ratio = torch.exp(log_prob - rollout_data.old_log_prob)

#             # clipped surrogate loss
#             policy_loss_1 = advantages * ratio
#             policy_loss_2 = advantages * torch.clamp(
#                 ratio, 1 - clip_range, 1 + clip_range
#             )
#             policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

#             # Logging
#             pg_losses.append(policy_loss.item())
#             clip_fraction = torch.mean(
#                 (torch.abs(ratio - 1) > clip_range).float()
#             ).item()
#             clip_fractions.append(clip_fraction)

#             if self.clip_range_vf is None:
#                 # No clipping
#                 values_pred = values
#             else:
#                 # Clip the difference between old and new value
#                 # NOTE: this depends on the reward scaling
#                 values_pred = rollout_data.old_values + torch.clamp(
#                     values - rollout_data.old_values, -clip_range_vf, clip_range_vf
#                 )
#             # Value loss using the TD(gae_lambda) target
#             value_loss = F.mse_loss(rollout_data.returns, values_pred)
#             value_losses.append(value_loss.item())

#             # Entropy loss favor exploration
#             if entropy is None:
#                 # Approximate entropy when no analytical form
#                 entropy_loss = -torch.mean(-log_prob)
#             else:
#                 entropy_loss = -torch.mean(entropy)

#             entropy_losses.append(entropy_loss.item())
#             loss = (
#                 policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
#             )

#             # Calculate approximate form of reverse KL Divergence for early stopping
#             # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
#             # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
#             # and Schulman blog: http://joschu.net/blog/kl-approx.html
#             with torch.no_grad():
#                 log_ratio = log_prob - rollout_data.old_log_prob
#                 approx_kl_div = (
#                     torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
#                 )
#                 approx_kl_divs.append(approx_kl_div)

#             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
#                 continue_training = False
#                 if self.verbose >= 1:
#                     print(
#                         f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
#                     )
#                 break

#             # Optimization step
#             self.policy.optimizer.zero_grad()
#             loss.backward()
#             # Clip grad norm
#             torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
#             self.policy.optimizer.step()

#         self._n_updates += 1
#         if not continue_training:
#             break

#     explained_var = explained_variance(
#         self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
#     )

#     # Logs
#     self.logger.record("train/entropy_loss", np.mean(entropy_losses))
#     self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
#     self.logger.record("train/value_loss", np.mean(value_losses))
#     self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
#     self.logger.record("train/clip_fraction", np.mean(clip_fractions))
#     self.logger.record("train/loss", loss.item())
#     self.logger.record("train/explained_variance", explained_var)
#     if hasattr(self.policy, "log_std"):
#         self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

#     self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
#     self.logger.record("train/clip_range", clip_range)
#     if self.clip_range_vf is not None:
#         self.logger.record("train/clip_range_vf", clip_range_vf)


class PCGradMultiEnvPPO(MultiEnvPPO):
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            grad_list = []
            # Do a complete pass on the rollout buffer
            for rollout_data_all_classes in self.rollout_buffer.get(self.batch_size):
                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                for rollout_data in rollout_data_all_classes:
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = self.policy.evaluate_actions(
                        rollout_data.observations, actions
                    )
                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(
                        ratio, 1 - clip_range, 1 + clip_range
                    )
                    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = torch.mean(
                        (torch.abs(ratio - 1) > clip_range).float()
                    ).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + torch.clamp(
                            values - rollout_data.old_values,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -torch.mean(-log_prob)
                    else:
                        entropy_loss = -torch.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = (
                        policy_loss
                        + self.ent_coef * entropy_loss
                        + self.vf_coef * value_loss
                    )

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = (
                            torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                            .cpu()
                            .numpy()
                        )
                        approx_kl_divs.append(approx_kl_div)

                    if (
                        self.target_kl is not None
                        and approx_kl_div > 1.5 * self.target_kl
                    ):
                        continue_training = False
                        if self.verbose >= 1:
                            print(
                                f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                            )
                        break

                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    grad_list.append(get_param_grad(self.policy.parameters()))

                pc_grad_list = proj_grad(grad_list)
                grad = param_sum(pc_grad_list)
                set_param_grad(self.policy.parameters(), grad)

                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
