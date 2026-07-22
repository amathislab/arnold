import numpy as np
import torch
from typing import NamedTuple, Generator
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from typing import Optional
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import TensorDict


class ExpertRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    expert_actions: torch.Tensor
    action_masks: torch.Tensor


class ExpertDictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    expert_actions: torch.Tensor
    action_masks: torch.Tensor


class ExpertRolloutBuffer(RolloutBuffer):
    """Rollout buffer that also stores the expert actions."""

    def reset(self):
        self.expert_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.action_masks = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=bool
        )
        super().reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, expert_action):
        self.expert_actions[self.pos] = np.array(expert_action)
        self.action_masks[self.pos] = np.array(action)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "expert_actions",
                "action_masks"
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env=None,
    ):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds],
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.expert_actions[batch_inds],
            self.action_masks[batch_inds],
        )
        return ExpertRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class ExpertDictRolloutBuffer(DictRolloutBuffer):
    def reset(self):
        self.expert_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32
        )
        self.action_masks = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=bool
        )
        super().reset()

    def add(self, obs, action, reward, episode_start, value, log_prob, expert_action, action_mask):
        self.expert_actions[self.pos] = np.array(expert_action)
        self.action_masks[self.pos] = np.array(action_mask)
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[ExpertDictRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = [
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "expert_actions",
                "action_masks",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ExpertDictRolloutBufferSamples:
        return ExpertDictRolloutBufferSamples(
            observations={
                key: self.to_torch(obs[batch_inds])
                for (key, obs) in self.observations.items()
            },
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            expert_actions=self.to_torch(self.expert_actions[batch_inds]),
            action_masks=self.to_torch(self.action_masks[batch_inds]),
        )
