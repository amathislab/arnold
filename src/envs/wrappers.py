import numpy as np
import torch
from gymnasium.core import ObservationWrapper, ActionWrapper
from gymnasium import spaces
from typing import Dict
from definitions import MASK_SUFFIX
from envs.multi_env_utilites import fill_larger_box, compute_box_mask, compute_dict_mask


class PaddedObservationWrapper(ObservationWrapper):
    """Class to make the observation compatible with a larger observation space. This implementation
    returns the observation as a numpy array (or as a dictionary of numpy arrays)"""

    def __init__(self, env, observation_space, ignore_last_dim=True):
        super().__init__(env)
        self._check_space_compatibility(observation_space)
        self._observation_space = observation_space
        self.random_obs = observation_space.sample()
        self.ignore_last_dim = ignore_last_dim
        self.obs_mask = self.compute_mask()

    def _check_space_compatibility(self, observation_space):
        if isinstance(observation_space, spaces.Dict):
            for key, space in self.env.observation_space.items():
                if not isinstance(space, spaces.Box):
                    raise NotImplementedError("Only implemented for Box or Dict[Box]")
                target_space = observation_space.get(key)
                if target_space is None:
                    raise ValueError(
                        "Space incompatibility: the target space does not have the key ",
                        key,
                    )
                for dim, target_dim in zip(space.shape, target_space.shape):
                    if dim > target_dim:
                        raise ValueError(
                            "Incompatible spaces for key ", key, space, target_space
                        )
        elif isinstance(observation_space, spaces.Box):
            for dim, target_dim in zip(
                self.env.observation_space.shape, observation_space.shape
            ):
                if dim > target_dim:
                    raise ValueError(
                        "Incompatible spaces",
                        self.env.observation_space,
                        observation_space,
                    )

    def observation(self, observation):
        if isinstance(observation, np.ndarray) or isinstance(observation, torch.Tensor):
            obs = self.random_obs.copy()
            obs = fill_larger_box(obs, observation, self.obs_mask, self.ignore_last_dim)
            return obs
        elif isinstance(observation, Dict):
            obs = {key: val.copy() for key, val in self.random_obs.items()}
            for key, value in observation.items():
                obs[key] = fill_larger_box(
                    obs[key],
                    value,
                    self.obs_mask[key + MASK_SUFFIX],
                    self.ignore_last_dim,
                )
            obs.update(self.obs_mask)
            return obs

    def compute_mask(self):
        # create the mask comparing _observation_space and env.observation_space
        if isinstance(self.observation_space, spaces.Box):
            return compute_box_mask(
                self.observation_space, self.env.observation_space, self.ignore_last_dim
            )
        elif isinstance(self.observation_space, spaces.Dict):
            return compute_dict_mask(
                self.observation_space, self.env.observation_space, self.ignore_last_dim
            )
        else:
            raise NotImplementedError(
                "Only implemented for Box or Dict of Box", self.observation_space
            )


class PaddedActionWrapper(ActionWrapper):
    def __init__(self, env, action_space):
        super().__init__(env)
        self._check_space_compatibility(action_space)
        self._action_space = action_space
        self.random_action = action_space.sample()
        self.action_mask = self.compute_mask()

    def _check_space_compatibility(self, action_space):
        if isinstance(action_space, spaces.Box):
            for dim, target_dim in zip(action_space.shape, self.env.action_space.shape):
                if dim < target_dim:
                    raise ValueError(
                        "Incompatible action spaces",
                        action_space,
                        self.env.action_space,
                    )
        else:
            raise NotImplementedError(
                "The current implementation requires a Box action space"
            )

    def action(self, padded_action):
        unpadded_action = padded_action[~self.action_mask].reshape(
            self.env.action_space.shape
        )
        return unpadded_action

    def compute_mask(self):
        # create the mask comparing _action_space and env.action_space
        if isinstance(self.action_space, spaces.Box):
            mask = compute_box_mask(
                self.action_space, self.env.action_space, ignore_last_dim=False
            )
            return mask
        else:
            raise NotImplementedError(
                "Only implemented for Box or Dict of Box", self.action_space
            )
