import gym
import numpy as np
from collections import deque
from definitions import OBS_KEY, OBS_ID_KEY, ACTION_ID_KEY, PADDING_KEY
from vocabulary import VOCABULARY


class SpecsObsMixin:
    """This class enhances the observation by including a history of transitions and the specification of
    each action component."""

    def _specs_obs_init_addon(self, include_adapt_state, num_memory_steps):
        """Function to augment an environment's init function by changing
        the observation space and the containers for the transition history.

        Args:
            include_adapt_state (bool): whether to include the transition history
            num_memory_steps (int): number of transitions to include in the state
        """
        self.obs_ids = self.get_obs_ids()
        self.action_ids = self.get_action_ids()
        self._include_adapt_state = include_adapt_state
        if include_adapt_state:
            self.num_memory_steps = num_memory_steps
            self._obs_prev_list = deque(
                [], maxlen=num_memory_steps + 1
            )  # Storing observations from newest to oldest
        else:
            self.num_memory_steps = 0

        single_obs = self.get_obs()
        obs_box = gym.spaces.Box(
            -100_000, 100_000, shape=(*np.shape(single_obs), 1 + self.num_memory_steps)
        )
        obs_ids_box = gym.spaces.Box(-100_000, 100_000, shape=self.obs_ids.shape)
        action_ids_box = gym.spaces.Box(-100_000, 100_000, shape=self.action_ids.shape)
        obs_space = gym.spaces.Dict(
            {OBS_KEY: obs_box, OBS_ID_KEY: obs_ids_box, ACTION_ID_KEY: action_ids_box}
        )
        self.observation_space = obs_space

    def create_history_reset_obs(self, obs):
        """Function to augment the state returned by the reset of a myosuite environment

        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the keys specified in self.obs_keys, if dict mode, otherwise an array with
            the values of the dict
        """
        if self._include_adapt_state:
            self.clear_history()
        return self.create_history_obs(obs)

    def create_history_obs(self, obs):
        """Function to augment the state returned by the step function of a myosuite environment
        Args:
            obs_dict (np.array): current observation

        Returns:
            dict with the keys OBS_KEY, OBS_ID_KEY, ACTION_ID_KEY
        """
        if self._include_adapt_state:
            self._obs_prev_list.append(obs)
            obs_val = np.stack(self._obs_prev_list, axis=-1)
        else:
            obs_val = obs

        return_dict = {
            OBS_KEY: obs_val,
            OBS_ID_KEY: self.obs_ids,
            ACTION_ID_KEY: self.action_ids,
        }
        return return_dict

    def get_obs_ids(self):
        obs_specs_dict = self.get_obs_ids_dict()

        # Assume that all the specs under the same key have the same length
        max_specs_len = max([len(s[0]) for s in obs_specs_dict.values()])
        obs_spec_per_key_list = []
        for obs_key in self.obs_keys:
            key_specs = obs_specs_dict[obs_key]

            # Set the initial value to that of the padding key
            key_ids = VOCABULARY[PADDING_KEY] * np.ones((len(key_specs), max_specs_len))
            for key_obs_idx, spec_list in enumerate(key_specs):
                for spec_idx, spec in enumerate(spec_list):
                    spec_id = VOCABULARY[spec]
                    key_ids[key_obs_idx, spec_idx] = spec_id
            obs_spec_per_key_list.append(key_ids.astype(np.float32))
        obs_specs = np.concatenate(obs_spec_per_key_list, axis=0)
        return obs_specs

    def clear_history(self):
        # Empty the history cache and store the new obs
        self._obs_prev_list.clear()
        zero_obs = np.zeros(self.observation_space[OBS_KEY].shape[0], dtype=np.float32)
        for _ in range(self.num_memory_steps):
            self._obs_prev_list.append(zero_obs)
