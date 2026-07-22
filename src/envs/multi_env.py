import copy
import inspect
import gymnasium as gym
import multiprocessing as mp
import numpy as np
import pickle
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper, VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import _worker
from stable_baselines3.common.preprocessing import is_image_space
from typing import List, Callable, Optional
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from envs.running_mean_std import RunningMeanStdFloat32, SpecsRunningMeanStd
from envs.wrappers import PaddedActionWrapper, PaddedObservationWrapper
from definitions import MASK_SUFFIX, OBS_KEY, OBS_ID_KEY
from typing import Union, Dict, Any, Tuple
from copy import deepcopy
from stable_baselines3.common.running_mean_std import RunningMeanStd
from envs.multi_env_utilites import (
    merge_observation_spaces,
    merge_action_spaces,
    ForkedPdb,
)


def wrap_with_padding(env_fn, observation_space, action_space, ignore_last_dim):
    env = env_fn()
    act_padded_env = PaddedActionWrapper(env, action_space)
    obs_act_padded_env = PaddedObservationWrapper(
        act_padded_env, observation_space, ignore_last_dim
    )
    return obs_act_padded_env


class MyVecNormalize(VecNormalize):

    def set_venv(self, venv: VecEnv) -> None:
        """
        Sets the vector environment to wrap to venv.

        Also sets attributes derived from this such as `num_env`.

        :param venv:
        """
        if self.venv is not None:
            raise ValueError(
                "Trying to set venv of already initialized VecNormalize wrapper."
            )
        self.venv = venv
        self.num_envs = venv.num_envs
        self.class_attributes = dict(inspect.getmembers(self.__class__))
        self.render_mode = venv.render_mode

        # Check that the observation_space shape match
        # utils.check_shape_equal(self.observation_space, venv.observation_space)
        self.returns = np.zeros(self.num_envs)

    @staticmethod
    def load(load_path: str, venv: VecEnv) -> "MyVecNormalize":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)
        vec_normalize.set_venv(venv)
        return vec_normalize


class RewardNormalizer(gym.Wrapper[ObsType, ActType, ObsType, ActType]):

    def __init__(
        self,
        env: VecEnv,
        vec_normalize: VecNormalize,
        current_env_id: int,
    ):
        self.env = env
        self.vec_normalize = vec_normalize
        self.env_id = current_env_id
        print(
            f"Env: {type(env)}, Id: {current_env_id}, RewardNormalizer: {type(vec_normalize)}"
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(self.vec_normalize, FlexibleMultiVecNormalize):
            norm_reward = self.vec_normalize.normalize_single_reward(
                reward, self.env_id
            ).item()
        elif isinstance(self.vec_normalize, VecNormalize):
            norm_reward = self.vec_normalize.normalize_reward(reward)
        else:
            raise NotImplementedError(
                f"VecNormalize type {type(self.vec_normalize)} is not supported"
            )
        # print(f"Env: {type(self.env)}, RewardNormalizer: {type(self.vec_normalize)}, RewardShape: {reward.shape}, NormRewardShape: {norm_reward.shape}")
        return obs, norm_reward, terminated, truncated, info


class EnvIDWrapper(gym.ObservationWrapper):

    def __init__(self, env: VecEnv, env_id: int):
        super().__init__(env)
        self.env_id = env_id
        self._observation_space = copy.deepcopy(env.observation_space)
        self._observation_space.spaces["env_id"] = spaces.Box(
            low=0, high=100, shape=(1, 1), dtype=np.int32
        )
        print(f"Env: {type(env)}, Id: {env_id}, Append Env Id")

    # def step(self, action: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    #     obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
    #     if isinstance(obs, dict) :
    #         obs["env_id"] = np.array([self.env_id])
    #     else :
    #         raise NotImplementedError(f"Observation type {type(obs)} is not supported")
    #     return obs, reward, terminated, truncated, info

    # def reset(self, **kwargs):
    #     obs = self.env.reset(**kwargs)
    #     if isinstance(obs, dict) :
    #         obs["env_id"] = np.array([self.env_id])
    #     else :
    #         raise NotImplementedError(f"Observation type {type(obs)} is not supported")
    #     return obs

    def observation(self, observation):
        if isinstance(observation, dict):
            observation["env_id"] = np.array([[self.env_id]])
        else:
            raise NotImplementedError(
                f"Observation type {type(observation)} is not supported"
            )
        return observation


class MultiSubprocVecEnv(SubprocVecEnv):
    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        start_method: Optional[str] = None,
        mask_last_obs_dim=False,
    ):
        self.waiting = False
        self.closed = False
        self.mask_last_obs_dim = mask_last_obs_dim
        self.n_envs = len(env_fns)

        # I do not see any better way than instantiating all the environments and then deleting
        # them, although it's not the most elegant solution it works
        observation_spaces = []
        action_spaces = []
        for env_fn in env_fns:
            env = env_fn()
            observation_spaces.append(env.observation_space)
            action_spaces.append(env.action_space)

        observation_space = merge_observation_spaces(
            observation_spaces, mask_last_obs_dim
        )
        action_space = merge_action_spaces(action_spaces)

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            wrapped_env_fn = CloudpickleWrapper(
                lambda: wrap_with_padding(
                    env_fn,
                    observation_space,
                    action_space,
                    ignore_last_dim=not mask_last_obs_dim,
                )
            )
            args = (work_remote, remote, wrapped_env_fn)
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # type: ignore[attr-defined]
            process.start()
            self.processes.append(process)
            work_remote.close()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    @staticmethod
    def get_larger_box(box_list):
        dim_list = []
        for box in box_list:
            for i, s in enumerate(box.shape):
                if len(dim_list) <= i:
                    dim_list.append(s)
                else:
                    dim_list[i] = max(dim_list[i], s)

        low = np.min([np.min(b.low) for b in box_list])
        high = np.max([np.max(b.high) for b in box_list])
        return spaces.Box(low, high, shape=dim_list)


class FlexibleMultiVecNormalize(VecNormalize):
    """Env normalizer thought to work with Arnold. It accepts observations
    from different environments and tries to normalize the components which
    come from the same sensor in the same way. For this reason, it only works
    with environments which provide this information (same prerequisite as
    the Arnold policy network). The observation should be a dictionary with
    the following keys: {OBS_KEY, OBS_ID_KEY, OBS_KEY + OBS_MASK_SUFFIX}. It only
    normalizes the input corresponding to OBS_KEY and assumes OBS_ID_KEY to contain
    a vector for each observation component, while "obs_mask" is a bool mask of
    the observation.
    """

    def __init__(
        self,
        venv: VecEnv,
        training: bool = True,
        norm_obs: bool = True,
        norm_reward: bool = True,
        clip_obs: float = 10,
        clip_reward: float = 10,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        VecEnvWrapper.__init__(self, venv)
        self.norm_obs = norm_obs  # Store the parameter
        self.env_ids_vec = self.venv.get_attr("id")
        self.env_id_set = set(
            self.env_ids_vec
        )  # unique env ids currently part of this vecnormalize
        self.env_id_mask_dict = {
            env_id: np.array([(e == env_id) for e in self.env_ids_vec])
            for env_id in self.env_id_set
        }
        # for reward normalization
        self.ret_rms_per_id = {
            env_id: RunningMeanStdFloat32(shape=()) for env_id in self.env_id_set
        }
        if self.norm_obs:  # Only initialize observation normalization if enabled
            self._sanity_checks()
            # for observation normalization
            obs_id_signatures = venv.env_method("get_obs_ids")
            obs_mask_list = [
                o[OBS_KEY + MASK_SUFFIX] for o in venv.get_attr(OBS_KEY + MASK_SUFFIX)
            ]
            obs_mask = ~np.array(obs_mask_list)
            self.obs_rms = SpecsRunningMeanStd(
                obs_shape=self.observation_space[OBS_KEY].shape,
                obs_id_signatures=obs_id_signatures,
                obs_mask=obs_mask,
            )
        self.ret_rms = RunningMeanStd(shape=())
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        # Returns: discounted rewards
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.training = training
        self.norm_reward = norm_reward
        self.old_reward = np.array([])

    def _sanity_checks(self) -> None:
        if not isinstance(self.observation_space, spaces.Dict):
            raise ValueError("Only works with dict observation spaces")

        for key in [OBS_KEY, OBS_ID_KEY, OBS_KEY + MASK_SUFFIX]:
            if key not in self.observation_space.keys():
                raise ValueError("The observation must inclue ", key)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Restores pickled state.

        User must call set_venv() after unpickling before using.

        :param state:"""
        self.__dict__.update(state)
        assert "venv" not in state
        self.venv = None  # type: ignore[assignment]

    def set_venv(self, venv: VecEnv, old_vocabulary=None) -> None:
        if self.venv is not None:
            raise ValueError(
                "Trying to set venv of already initialized VecNormalize wrapper."
            )
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.class_attributes = dict(inspect.getmembers(self.__class__))
        self.render_mode = venv.render_mode

        self.returns = np.zeros(self.num_envs)

        # Update the reward normalization statistics
        self.env_ids_vec = self.venv.get_attr("id")
        self.env_id_set = set(self.env_ids_vec)

        # Old version had one single normalization, keep for compatibility
        if not hasattr(self, "ret_rms_per_id"):
            assert not hasattr(self, "env_id_mask_dict"), "Old ver. should not have it"
            assert hasattr(self, "ret_rms"), "Old ver. should have it"
            self.ret_rms_per_id = {
                env_id: self.ret_rms.copy() for env_id in self.env_id_set
            }
        else:
            for env in self.env_id_set:
                if env not in self.ret_rms_per_id:
                    # Then it's a new env and we should init its reward params
                    self.ret_rms_per_id[env] = RunningMeanStdFloat32(shape=())
        # Either way update the env_id_mask_dict
        self.env_id_mask_dict = {
            env_id: np.array([(e == env_id) for e in self.env_ids_vec])
            for env_id in self.env_id_set
        }

        if self.norm_obs:
            # Update the known signatures with the ones of the new venv
            obs_id_signatures = venv.env_method("get_obs_ids")

            # Save the old obs_rms and add space for the new signatures
            old_obs_rms = self.obs_rms
            obs_mask_list = [
                o[OBS_KEY + MASK_SUFFIX] for o in venv.get_attr(OBS_KEY + MASK_SUFFIX)
            ]
            obs_mask = ~np.array(obs_mask_list)

            self.obs_rms = SpecsRunningMeanStd(
                obs_shape=venv.observation_space[OBS_KEY].shape,
                obs_id_signatures=obs_id_signatures,
                obs_mask=obs_mask,
            )
            self.obs_rms.combine(old_obs_rms, old_vocabulary)

    def step_wait(self) -> VecEnvStepReturn:
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, dones)

        where ``dones`` is a boolean vector indicating whether each element is new.
        """
        obs, rewards, dones, infos = self.venv.step_wait()
        self.old_obs = obs
        self.old_reward = rewards

        if self.training and self.norm_obs:
            self.obs_rms.update(obs[OBS_KEY])

        obs = self.normalize_obs(obs)

        if self.training:
            self._update_reward(rewards)
        rewards = self.normalize_reward(rewards)

        # Normalize the terminal observations
        for env_idx, done in enumerate(dones):
            if done and "terminal_observation" in infos[env_idx]:
                terminal_obs = infos[env_idx]["terminal_observation"]
                terminal_obs[OBS_KEY] = self.normalize_single_obs(
                    terminal_obs[OBS_KEY], env_idx=env_idx
                )

        self.returns[dones] = 0
        return obs, rewards, dones, infos

    def _update_reward(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.returns = self.returns * self.gamma + reward
        for env_id in self.env_id_set:
            ret_rms = self.ret_rms_per_id[env_id]
            mask = self.env_id_mask_dict[env_id]
            ret_rms.update(self.returns[mask])

    def _unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs * np.sqrt(self.obs_rms.var + self.epsilon)) + self.obs_rms.mean

    def _obs_mean_var(self, obs_data: np.ndarray):
        """Return (mean, var) broadcastable against obs_data.

        ``obs_rms.mean`` has shape ``(num_envs, tokens, timesteps)``.  We need
        to handle three calling conventions:
          - ndim == 2  →  single-env terminal obs ``(tokens, timesteps)``
          - ndim == 3, shape[0] == num_envs  →  normal rollout batch
          - ndim == 3, shape[0] != num_envs  →  off-policy replay batch
        """
        if obs_data.ndim == 2 or obs_data.shape[0] != self.num_envs:
            keepdims = obs_data.ndim == 3
            mean = self.obs_rms.mean.mean(axis=0, keepdims=keepdims)
            var = self.obs_rms.var.mean(axis=0, keepdims=keepdims)
        else:
            mean = self.obs_rms.mean
            var = self.obs_rms.var
        return mean, var

    def normalize_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize observations using this VecNormalize's observations statistics.
        Calling this method does not update statistics.

        Handles three calling conventions (see ``_obs_mean_var`` for details):
        normal rollout, single-env terminal obs, and off-policy replay batch.
        """
        obs_ = deepcopy(obs)
        if self.norm_obs:
            obs_data = obs[OBS_KEY]
            mean, var = self._obs_mean_var(obs_data)
            obs_[OBS_KEY] = np.clip(
                (obs_data - mean) / np.sqrt(var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
        return obs_

    def normalize_single_obs_dict(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """
        Normalize a single dict observation using this VecNormalize's observations statistics.
        Calling this method does not update statistics.
        """
        obs_ = deepcopy(obs)
        if self.norm_obs:
            obs_[OBS_KEY] = np.clip(
                (obs[OBS_KEY] - self.obs_rms.mean_single_env(env_idx))
                / np.sqrt(self.obs_rms.var_single_env(env_idx) + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
        return obs_

    def normalize_single_obs(self, obs: np.ndarray, env_idx: int) -> np.ndarray:
        """
        Normalize a single observation array using this VecNormalize's statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = np.clip(
                (obs - self.obs_rms.mean_single_env(env_idx))
                / np.sqrt(self.obs_rms.var_single_env(env_idx) + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
        return obs

    def unnormalize_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        obs_ = deepcopy(obs)
        if self.norm_obs:
            obs_data = obs[OBS_KEY]
            mean, var = self._obs_mean_var(obs_data)
            obs_[OBS_KEY] = (obs_data * np.sqrt(var + self.epsilon)) + mean
        return obs_

    def normalize_reward(self, reward: np.ndarray) -> np.ndarray:
        """
        Normalize rewards using this VecNormalize's rewards statistics.
        Calling this method does not update statistics.
        """
        norm_reward = reward.copy()
        if self.norm_reward:
            n_envs = len(self.env_ids_vec)
            if reward.ndim == 1 and reward.shape[0] == n_envs:
                # Step-time call: shape (n_envs,) — apply per-env rms via mask
                for env_id in self.env_id_set:
                    ret_rms = self.ret_rms_per_id[env_id]
                    mask = self.env_id_mask_dict[env_id]
                    norm_reward[mask] = np.clip(
                        norm_reward[mask] / np.sqrt(ret_rms.var + self.epsilon),
                        -self.clip_reward,
                        self.clip_reward,
                    )
            else:
                # Replay buffer sampling call: shape (batch_size, 1) — env identity
                # is unknown, so normalize with the mean variance across all env ids
                mean_var = float(np.mean([rms.var for rms in self.ret_rms_per_id.values()]))
                norm_reward = np.clip(
                    norm_reward / np.sqrt(mean_var + self.epsilon),
                    -self.clip_reward,
                    self.clip_reward,
                )
        return norm_reward

    def normalize_single_reward(self, reward: np.ndarray, env_id: int) -> np.ndarray:
        norm_reward = reward.copy()
        if self.norm_reward:
            for env_name in self.env_id_set:
                ret_rms = self.ret_rms_per_id[env_name]
                mask = self.env_id_mask_dict[env_name][env_id]
                # print(env_id, self.env_id_mask_dict[env_name])
                if mask:
                    norm_reward = np.clip(
                        norm_reward[..., mask] / np.sqrt(ret_rms.var + self.epsilon),
                        -self.clip_reward,
                        self.clip_reward,
                    )
        return norm_reward

    def unnormalize_reward(self, reward: np.ndarray) -> np.ndarray:
        unnorm_reward = reward.copy()
        if self.norm_reward:
            n_envs = len(self.env_ids_vec)
            if reward.ndim == 1 and reward.shape[0] == n_envs:
                for env_id in self.env_id_set:
                    ret_rms = self.ret_rms_per_id[env_id]
                    mask = self.env_id_mask_dict[env_id]
                    unnorm_reward[mask] = unnorm_reward[mask] * np.sqrt(ret_rms.var + self.epsilon)
            else:
                mean_var = float(np.mean([rms.var for rms in self.ret_rms_per_id.values()]))
                unnorm_reward = unnorm_reward * np.sqrt(mean_var + self.epsilon)
        return unnorm_reward

    def unnormalize_single_reward(self, reward: np.ndarray, env_id: int) -> np.ndarray:
        unnorm_reward = reward.copy()
        if self.norm_reward:
            for env_name in self.env_id_set:
                ret_rms = self.ret_rms_per_id[env_name]
                mask = self.env_id_mask_dict[env_name][env_id]
                if mask:
                    unnorm_reward = unnorm_reward * np.sqrt(ret_rms.var + self.epsilon)
        return unnorm_reward

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        :return: first observation of the episode
        """
        obs = self.venv.reset()
        self.old_obs = obs
        self.returns = np.zeros(self.num_envs)
        if self.training and self.norm_obs:
            self.obs_rms.update(obs[OBS_KEY])
        return self.normalize_obs(obs)

    @staticmethod
    def load(load_path: str, venv: VecEnv, vocabulary: Dict = None) -> "VecNormalize":
        """
        Loads a saved VecNormalize object.

        :param load_path: the path to load from.
        :param venv: the VecEnv to wrap.
        :param vocabulary: the vocabulary used by the vecnormalize to load.
        :return:
        """
        with open(load_path, "rb") as file_handler:
            vec_normalize = pickle.load(file_handler)

        # Backward compatibility
        if "norm_obs" not in vec_normalize.__dict__.keys():
            vec_normalize.norm_obs = True

        vec_normalize.set_venv(venv, old_vocabulary=vocabulary)
        return vec_normalize
