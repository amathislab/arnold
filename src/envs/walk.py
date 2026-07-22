import numpy as np
import collections
from myosuite.envs.myo.walk_v0 import ReachEnvV0, WalkEnvV0
from definitions import ACT_KEY, GOAL_KEY
from envs.env_mixins import SpecsObsMixin, ObsEmbeddingMixin


class MuscleLegsReachEnv(ReachEnvV0, SpecsObsMixin, ObsEmbeddingMixin):
    OBS_KEYS = [ACT_KEY, GOAL_KEY]
    CUSTOM_RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "bonus": 4.0,
        "penalty": 50,
        "act_reg": 1,
        "alive": 5,
    }

    def __init__(
        self,
        model_path,
        obsd_model_path=None,
        seed=None,
        include_adapt_state=False,
        num_memory_steps=30,
        **kwargs,
    ):
        self._init_done = False
        super().__init__(
            model_path, obsd_model_path=obsd_model_path, seed=seed, **kwargs
        )
        self.action_dim = self.sim.model.nu
        self._specs_obs_init_addon(include_adapt_state, num_memory_steps)
        self._obs_embedding_init_addon()
        self._init_done = True

    def _setup(
        self,
        target_reach_range: dict,
        far_th=0.35,
        obs_keys: list = OBS_KEYS,
        weighted_reward_keys: dict = CUSTOM_RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
        super()._setup(
            target_reach_range=target_reach_range,
            far_th=far_th,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        self.obs_keys.remove("act")

    def reset(self):
        super().reset()
        obs = self.create_history_reset_obs(self.obs_dict)
        obs = self.add_positions_to_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(info.get("rwd_dict"))
        if self._init_done:
            obs = self.create_history_obs(self.obs_dict)
            obs = self.add_positions_to_obs(obs)
        return obs, reward, done, info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["muscle_len"] = np.nan_to_num(sim.data.actuator_length.copy())
        obs_dict["muscle_vel"] = np.nan_to_num(sim.data.actuator_velocity.copy())
        obs_dict["muscle_force"] = np.nan_to_num(sim.data.actuator_force.copy())

        muscle_keys = ("muscle_len", "muscle_vel", "muscle_force", "act")
        obs_dict["actuator_obs"] = np.row_stack(
            [obs_dict[key] for key in muscle_keys]
        )  # num_channels * num_actuators

        goal_keys = ("reach_err",)
        # num_channels * num_goals = 1 * 15
        obs_dict["goal_obs"] = np.row_stack([obs_dict[key] for key in goal_keys])
        return obs_dict

    def get_reward_dict(self, obs_dict):
        reach_dist = np.linalg.norm(obs_dict["reach_err"], axis=-1)
        vel_dist = np.linalg.norm(obs_dict["qvel"], axis=-1)
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )
        far_th = (
            self.far_th * len(self.tip_sids)
            if np.squeeze(obs_dict["time"]) > 2 * self.dt
            else np.inf
        )
        # near_th = len(self.tip_sids)*.0125
        near_th = len(self.tip_sids) * 0.050
        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("reach", -1.0 * reach_dist - 10.0 * vel_dist),
                (
                    "bonus",
                    1.0 * (reach_dist < 2 * near_th) + 1.0 * (reach_dist < near_th),
                ),
                ("act_reg", -100.0 * act_mag),
                ("alive", reach_dist <= far_th),
                ("penalty", -1.0 * (reach_dist > far_th)),
                # Must keys
                ("sparse", -1.0 * reach_dist),
                ("solved", reach_dist < near_th),
                ("done", reach_dist > far_th),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def get_obs_elements(self):
        actuators = list(self.sim.model.actuator_names)
        objects = []
        goals = []
        for target in self.target_reach_range.keys():
            goals.extend([f"{target}_x", f"{target}_y", f"{target}_z"])
        return [*actuators, *objects, *goals]


class MuscleLegsWalkEnv(WalkEnvV0, SpecsObsMixin, ObsEmbeddingMixin):
    OBS_KEYS = [ACT_KEY]  # TODO: encode goal velocity, period, ...
    CUSTOM_RWD_KEYS_AND_WEIGHTS = {
        "vel_reward": 5.0,
        "done": -100,
        "cyclic_hip": -10,
        "ref_rot": 10.0,
        "joint_angle_rew": 5.0,
        "alive": 5,
    }

    def __init__(
        self,
        model_path,
        obsd_model_path=None,
        seed=None,
        include_adapt_state=False,
        num_memory_steps=30,
        **kwargs,
    ):
        self._init_done = False
        super().__init__(
            model_path, obsd_model_path=obsd_model_path, seed=seed, **kwargs
        )
        self.action_dim = self.sim.model.nu
        self._specs_obs_init_addon(include_adapt_state, num_memory_steps)
        self._obs_embedding_init_addon()
        self._init_done = True

    def _setup(
        self,
        obs_keys: list = OBS_KEYS,
        weighted_reward_keys: dict = CUSTOM_RWD_KEYS_AND_WEIGHTS,
        min_height=0.8,
        max_rot=0.8,
        hip_period=100,
        reset_type="init",
        target_x_vel=0.0,
        target_y_vel=1.2,
        target_rot=None,
        **kwargs,
    ):
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            min_height=min_height,
            max_rot=max_rot,
            hip_period=hip_period,
            reset_type=reset_type,
            target_x_vel=target_x_vel,
            target_y_vel=target_y_vel,
            target_rot=target_rot,
            **kwargs,
        )
        self.obs_keys.remove("act")

    def reset(self):
        super().reset()
        obs = self.create_history_reset_obs(self.obs_dict)
        obs = self.add_positions_to_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        info.update(info.get("rwd_dict"))
        if self._init_done:
            obs = self.create_history_obs(self.obs_dict)
            obs = self.add_positions_to_obs(obs)
        return obs, reward, done, info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        muscle_keys = ("muscle_length", "muscle_velocity", "muscle_force", "act")
        obs_dict[ACT_KEY] = np.row_stack(
            [obs_dict[key] for key in muscle_keys]
        )  # num_channels * num_actuators
        return obs_dict

    def get_reward_dict(self, obs_dict):
        vel_reward = self._get_vel_reward()
        cyclic_hip = self._get_cyclic_rew()
        ref_rot = self._get_ref_rotation_rew()
        joint_angle_rew = self._get_joint_angle_rew(
            ["hip_adduction_l", "hip_adduction_r", "hip_rotation_l", "hip_rotation_r"]
        )
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("vel_reward", vel_reward),
                ("cyclic_hip", cyclic_hip),
                ("ref_rot", ref_rot),
                ("joint_angle_rew", joint_angle_rew),
                ("act_mag", act_mag),
                # Must keys
                ("sparse", vel_reward),
                ("solved", vel_reward >= 1.0),
                ("done", self._get_done()),
            )
        )
        rwd_dict["alive"] = ~rwd_dict["done"]
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def get_obs_elements(self):
        actuators = list(self.sim.model.actuator_names)
        objects = []
        goals = []
        return [*actuators, *objects, *goals]
