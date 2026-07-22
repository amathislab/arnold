import imitation
import numpy as np
import torch
import torch.nn.functional as F
import pathlib
import io
import warnings
from typing import Dict
from gymnasium import spaces
from typing import Type, Union, Optional
from typing import Any, Dict, Optional, Type, Union
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    obs_as_tensor,
    get_system_info,
)
from stable_baselines3.ppo import PPO
from algos.buffers import ExpertRolloutBuffer, ExpertDictRolloutBuffer
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.base_class import SelfBaseAlgorithm
from stable_baselines3.common.vec_env.patch_gym import _convert_space
from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr, recursive_getattr
from models.ppo.ppo import TensorDict
from vocabulary import VOCABULARY
from definitions import MASK_SUFFIX


class BCPPO(PPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: Union[float, Schedule] = 0.0,
        vf_coef: Union[float, Schedule] = 0.5,
        pg_coef: Union[float, Schedule] = 1.0,
        imitation_coef: Union[float, Schedule] = 0.0,
        imitation_loss: str = "mse",
        use_expert_actions: bool = False,
        constant_loss_weight: bool = False,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        # Store the initial coefficients/schedules
        self.pg_coef_schedule = pg_coef
        self.imitation_coef_schedule = imitation_coef
        self.constant_loss_weight = constant_loss_weight
        self.ent_coef_schedule = ent_coef
        self.vf_coef_schedule = vf_coef

        # Initialize with starting values if schedules are provided
        self.pg_coef = pg_coef if isinstance(pg_coef, float) else pg_coef(1.0)
        self.imitation_coef = (
            imitation_coef if isinstance(imitation_coef, float) else imitation_coef(1.0)
        )
        self.ent_coef = ent_coef if isinstance(ent_coef, float) else ent_coef(1.0)
        self.vf_coef = vf_coef if isinstance(vf_coef, float) else vf_coef(1.0)
        self.imitation_loss = imitation_loss
        self.use_expert_actions = use_expert_actions

        self.num_different_envs = None

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=self.ent_coef,  # Pass the initial value
            vf_coef=self.vf_coef,  # Pass the initial value
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=None,
            rollout_buffer_kwargs=None,
            target_kl=target_kl,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def _setup_model(self) -> None:
        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = ExpertDictRolloutBuffer
            else:
                self.rollout_buffer_class = ExpertRolloutBuffer
        super()._setup_model()

    def collect_rollouts(
        self,
        env,
        callback,
        rollout_buffer: ExpertRolloutBuffer,
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

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # if self.num_different_envs is None:
            #     self.num_different_envs = (
            #         obs_tensor["env_id"].flatten().unique().shape[0]
            #     )

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
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            if self.imitation_coef > 0:
                expert_actions = np.zeros_like(clipped_actions, dtype=actions.dtype)
                action_mask = np.zeros_like(clipped_actions, dtype=bool)
                expert_actions_list = env.env_method("get_expert_action")
                for idx, expert_action in enumerate(expert_actions_list):
                    expert_actions[idx, : len(expert_action)] = expert_action
                    action_mask[idx, : len(expert_action)] = True
            else:
                expert_actions = clipped_actions
                action_mask = np.ones_like(clipped_actions, dtype=bool)

            if self.use_expert_actions:
                new_obs, rewards, dones, infos = env.step(expert_actions)
            else:
                new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
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
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            if isinstance(rollout_buffer, ExpertDictRolloutBuffer):
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                    expert_actions,
                    action_mask,
                )
            elif isinstance(rollout_buffer, ExpertRolloutBuffer):
                rollout_buffer.add(
                    self._last_obs,  # type: ignore[arg-type]
                    actions,
                    rewards,
                    self._last_episode_starts,  # type: ignore[arg-type]
                    values,
                    log_probs,
                    expert_actions,
                )
            else:
                raise ValueError("Invalid rollout buffer type")
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update coefficients based on remaining progress
        self._update_coefficients(self._current_progress_remaining)

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        imitation_losses = []
        clip_fractions = []
        # env_bc_losses = torch.zeros(self.num_different_envs, device=self.device, requires_grad=False)
        # env_bc_losses_count = torch.zeros(self.num_different_envs, device=self.device, requires_grad=False)

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                observations = rollout_data.observations
                if isinstance(observations, dict):
                    for key in observations.keys():
                        if key.endswith(MASK_SUFFIX):
                            observations[key] = observations[key].bool()
                actions = rollout_data.actions
                expert_actions = rollout_data.expert_actions
                action_masks = rollout_data.action_masks

                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()
                    expert_actions = expert_actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    observations, actions
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

                # Imitation loss
                if self.imitation_loss == "mse":
                    actions_pred, _, _ = self.policy(observations)
                    if self.constant_loss_weight:
                        masked_actions_pred = actions_pred[rollout_data.action_masks]
                        masked_expert_actions = expert_actions[
                            rollout_data.action_masks
                        ]
                        if isinstance(self.action_space, spaces.Discrete):
                            imitation_loss = F.cross_entropy(
                                masked_actions_pred, masked_expert_actions
                            )
                        else:
                            imitation_loss = F.mse_loss(
                                masked_actions_pred, masked_expert_actions
                            )
                    else:
                        sum_action_mask = action_masks.sum(dim=1)
                        invert_action_length = 1 / sum_action_mask
                        invert_action_length_mask = (
                            invert_action_length.unsqueeze(1) * action_masks
                        )
                        imitation_loss_full = F.mse_loss(
                            actions_pred, expert_actions, reduction="none"
                        )
                        imitation_loss_per_row = (
                            imitation_loss_full * invert_action_length_mask
                        ).sum(dim=1)
                        imitation_loss = imitation_loss_per_row.sum()

                elif self.imitation_loss == "neglogp":
                    values, log_prob, entropy = self.policy.evaluate_actions(
                        observations, expert_actions
                    )
                    imitation_loss_per_row = -log_prob
                    imitation_loss = imitation_loss_per_row.sum()

                # # env_bc_losses
                # env_id_index = observations["env_id"].flatten().long()
                # env_bc_losses.scatter_add_(
                #     0, env_id_index, imitation_loss_per_row
                # )
                # env_bc_losses_count.scatter_add_(0, env_id_index, torch.ones_like(env_id_index, dtype=env_bc_losses_count.dtype))
                imitation_losses.append(imitation_loss.item())

                loss = (
                    self.pg_coef * policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.imitation_coef * imitation_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # env_bc_losses /= env_bc_losses_count

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # for i in range(self.num_different_envs):
        #     self.logger.record(f"train/env_bc_loss_{i}", env_bc_losses[i].item())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/imitation_loss", np.mean(imitation_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record(
                "train/std", torch.exp(self.policy.log_std).mean().item()
            )

        # Log coefficients
        self.logger.record("train/pg_coef", self.pg_coef)
        self.logger.record("train/imitation_coef", self.imitation_coef)
        self.logger.record("train/ent_coef", self.ent_coef)
        self.logger.record("train/vf_coef", self.vf_coef)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def _update_coefficients(self, progress_remaining: float) -> None:
        """Update the current loss coefficients based on the remaining progress."""
        if not isinstance(self.pg_coef_schedule, float):
            self.pg_coef = self.pg_coef_schedule(progress_remaining)

        if not isinstance(self.imitation_coef_schedule, float):
            self.imitation_coef = self.imitation_coef_schedule(progress_remaining)

        if not isinstance(self.ent_coef_schedule, float):
            self.ent_coef = self.ent_coef_schedule(progress_remaining)

        if not isinstance(self.vf_coef_schedule, float):
            self.vf_coef = self.vf_coef_schedule(progress_remaining)

    @classmethod
    def load(
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        reset_std: bool = False,
        **kwargs,
    ) -> SelfBaseAlgorithm:
        model = super().load(
            path, env, device, custom_objects, print_system_info, force_reset, **kwargs
        )
        model._update_coefficients(model._current_progress_remaining)
        if reset_std:
            log_std_init = model.policy.log_std_init
            model.policy.log_std.data.fill_(log_std_init)
        return model

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[torch.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_iter: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, torch.optim.Optimizer):
                try:
                    attr.load_state_dict(params[name])  # type: ignore[arg-type]
                except ValueError:
                    print("WARNING: the set of trainable parameters has changed, "
                          "the optimizer state will not be restored.")
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )


class MultiTaskBCPPO(BCPPO):
    @classmethod
    def load(
        cls: Type[SelfBaseAlgorithm],
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        reset_std: bool = False,
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
        if (
            "policy_kwargs" in data
            and "features_extractor_kwargs" in data["policy_kwargs"]
        ):
            data["policy_kwargs"]["features_extractor_kwargs"]["vocabulary"] = (
                custom_objects.get("vocabulary")
            )
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()
        model.policy.to(model.device)

        try:
            # put state_dicts back in place
            model.set_parameters(params, exact_match=True, device=device)
            model.policy.set_vocabulary(VOCABULARY)
            if reset_std:
                log_std_init = model.policy.log_std_init
                model.policy.log_std.data.fill_(log_std_init)
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
                warnings.warn("Some state dict are not matching with the current model")
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

        # Update coefficients based on remaining progress
        model._update_coefficients(model._current_progress_remaining)

        return model

    def _update_current_progress_remaining(self, num_timesteps, total_timesteps):
        timesteps_since_restart = num_timesteps - self._num_timesteps_at_start
        timesteps_this_run = total_timesteps - self._num_timesteps_at_start
        self._current_progress_remaining = (
            1 - float(timesteps_since_restart) / float(timesteps_this_run)
        )
