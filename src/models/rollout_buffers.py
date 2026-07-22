import torch
import numpy as np
from stable_baselines3.common.buffers import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces
from typing import Union, Optional, Generator
from stable_baselines3.common.type_aliases import (
    DictRolloutBufferSamples,
)


class MultiEnvDictRolloutBuffer(DictRolloutBuffer):
    """IMPORTANT: here for every minibatch we are sampling batch_size * num_env_classes
    transitions. This is differetn from the non multi-env version, where we sample
    batch_size transitions.

    :param DictRolloutBuffer: _description_
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        num_env_classes: int = 1,
        num_envs_per_class: int = 1,
    ):
        self.num_env_classes = num_env_classes
        self.num_envs_per_class = num_envs_per_class
        assert n_envs == num_env_classes * num_envs_per_class
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=num_env_classes * num_envs_per_class,
        )

    def get(
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:
        indices = np.random.permutation(self.buffer_size * self.num_envs_per_class)

        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten_per_env(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten_per_env(
                    self.__dict__[tensor]
                )
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size

        start_idx = 0
        while start_idx < self.buffer_size * self.num_envs_per_class:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:
        samples_per_env = []
        for i in range(self.num_env_classes):
            sample = DictRolloutBufferSamples(
                observations={
                    key: self.to_torch(obs[i, batch_inds])
                    for (key, obs) in self.observations.items()
                },
                actions=self.to_torch(self.actions[i, batch_inds]),
                old_values=self.to_torch(self.values[i, batch_inds].flatten()),
                old_log_prob=self.to_torch(self.log_probs[i, batch_inds].flatten()),
                advantages=self.to_torch(self.advantages[i, batch_inds].flatten()),
                returns=self.to_torch(self.returns[i, batch_inds].flatten()),
            )
            samples_per_env.append(sample)
        return samples_per_env

    def swap_and_flatten_per_env(self, arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs), but keep the first
        index as the environment class index. It converts shape from [n_steps, n_envs, ...]
        (when ... is the shape of the features) to
        [n_env_classes, n_steps * n_envs_per_class, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(
            self.num_env_classes, shape[0] * self.num_envs_per_class, *shape[2:]
        )
