import collections
import numpy as np
from myosuite.envs.env_base import MujocoEnv
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.pen_v0 import PenTwirlRandomEnvV0
from myosuite.utils.quat_math import euler2quat
from myosuite.utils.vector_math import calculate_cosine
from envs.env_mixins import SpecsObsMixin
from vocabulary import VOCABULARY


class CustomPenEnv(PenTwirlRandomEnvV0):
    def get_reward_dict(self, obs_dict):
        pos_err = obs_dict["obj_err_pos"]
        pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = calculate_cosine(obs_dict["obj_rot"], obs_dict["obj_des_rot"])
        dropped = pos_align > 0.075
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )
        pos_align_diff = self.pos_align - pos_align  # should decrease
        rot_align_diff = rot_align - self.rot_align  # should increase
        alive = ~dropped

        rwd_dict = collections.OrderedDict(
            (
                # Optional Keys
                ("pos_align", -1.0 * pos_align),
                ("rot_align", rot_align),
                ("pos_align_diff", pos_align_diff),
                ("rot_align_diff", rot_align_diff),
                ("alive", alive),
                ("act_reg", -1.0 * act_mag),
                ("drop", -1.0 * dropped),
                (
                    "bonus",
                    1.0 * (rot_align > 0.9) * (pos_align < 0.075)
                    + 5.0 * (rot_align > 0.95) * (pos_align < 0.075),
                ),
                # Must keys
                ("sparse", -1.0 * pos_align + rot_align),
                ("solved", (rot_align > 0.95) * (~dropped)),
                ("done", dropped),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict

    def _setup(
        self,
        obs_keys: list = PenTwirlRandomEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = PenTwirlRandomEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        goal_orient_range=(
            -1,
            1,
        ),  # can be used to make the task simpler and limit the target orientations
        enable_rsi=False,
        rsi_distance=0,
        **kwargs,
    ):
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.S_grasp_sid = self.sim.model.site_name2id("S_grasp")
        self.obj_bid = self.sim.model.body_name2id("Object")
        self.eps_ball_sid = self.sim.model.site_name2id("eps_ball")
        self.obj_t_sid = self.sim.model.site_name2id("object_top")
        self.obj_b_sid = self.sim.model.site_name2id("object_bottom")
        self.tar_t_sid = self.sim.model.site_name2id("target_top")
        self.tar_b_sid = self.sim.model.site_name2id("target_bottom")
        self.pen_length = np.linalg.norm(
            self.sim.model.site_pos[self.obj_t_sid]
            - self.sim.model.site_pos[self.obj_b_sid]
        )
        self.tar_length = np.linalg.norm(
            self.sim.model.site_pos[self.tar_t_sid]
            - self.sim.model.site_pos[self.tar_b_sid]
        )

        self.goal_orient_range = goal_orient_range
        self.rsi = enable_rsi
        self.rsi_distance = rsi_distance
        self.pos_align = 0
        self.rot_align = 0

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        self.init_qpos[:-6] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.5  # place palm up

    def reset(self):
        # randomize target
        desired_orien = np.zeros(3)
        desired_orien[0] = self.np_random.uniform(
            low=self.goal_orient_range[0], high=self.goal_orient_range[1]
        )
        desired_orien[1] = self.np_random.uniform(
            low=self.goal_orient_range[0], high=self.goal_orient_range[1]
        )
        self.sim.model.body_quat[self.target_obj_bid] = euler2quat(desired_orien)

        if self.rsi:
            init_orien = np.zeros(3)
            init_orien[:2] = desired_orien[:2] + self.rsi_distance * (
                init_orien[:2] - desired_orien[:2]
            )
            self.sim.model.body_quat[self.obj_bid] = euler2quat(init_orien)

        self.robot.sync_sims(self.sim, self.sim_obsd)
        obs = MujocoEnv.reset(self)

        self.pos_align = np.linalg.norm(self.obs_dict["obj_err_pos"], axis=-1)
        self.rot_align = calculate_cosine(
            self.obs_dict["obj_rot"], self.obs_dict["obj_des_rot"]
        )

        return obs

    def step(self, a):
        obs, reward, done, info = super().step(a)
        self.pos_align = np.linalg.norm(self.obs_dict["obj_err_pos"], axis=-1)
        self.rot_align = calculate_cosine(
            self.obs_dict["obj_rot"], self.obs_dict["obj_des_rot"]
        )
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info

    def render(self, mode="human"):
        return self.sim.render(mode=mode)


class MusclePenEnv(CustomPenEnv, SpecsObsMixin):
    ALL_OBS_KEYS = [
        "hand_jnt",
        "hand_jnt_vel",
        "obj_pos",
        "obj_vel_lin",
        "obj_rot",
        "obj_velr",
        "obj_des_pos",
        "obj_des_rot",
        "obj_err_pos",
        "obj_err_rot",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "act",
    ]

    OBS_KEYS = [
        "hand_jnt",
        "hand_jnt_vel",
        "obj_pos",
        "obj_vel_lin",
        "obj_rot",
        "obj_velr",
        "obj_des_pos",
        "obj_des_rot",
        "obj_err_pos",
        "obj_err_rot",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "act",
    ]

    RWD_KEYS_AND_WEIGHTS = {
        "alive": 1.0,
        "solved": 1,
        "pos_align_diff": 100,
        "rot_align_diff": 100,
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
            model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            **kwargs,
        )
        self.action_dim = self.sim.model.nu
        self.muscle_names = [
            self.sim.model.id2name(i, "actuator") for i in range(self.sim.model.na)
        ]
        self.joint_names = [
            self.sim.model.id2name(i, "joint")
            for i in range(self.sim.model.nq - 6)  # 6 pen free joints
        ]
        self._specs_obs_init_addon(include_adapt_state, num_memory_steps)
        self._init_done = True

    def _setup(
        self,
        obs_keys: list = OBS_KEYS,
        weighted_reward_keys: list = RWD_KEYS_AND_WEIGHTS,
        goal_orient_range=(
            -1,
            1,
        ),  # can be used to make the task simpler and limit the target orientations
        enable_rsi=False,
        rsi_distance=0,
        **kwargs,
    ):
        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            goal_orient_range=goal_orient_range,
            enable_rsi=enable_rsi,
            rsi_distance=rsi_distance,
            **kwargs,
        )

    def reset(self):
        obs = super().reset().astype(np.float32)
        obs = self.create_history_reset_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self._init_done:
            obs = self.create_history_obs(obs.astype(np.float32))
            info.update(
                {f"{self.id}/{key}": val for key, val in info.get("rwd_dict").items()}
            )
        return obs, reward, done, info

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["hand_jnt_vel"] = sim.data.qvel[:-6].copy() * self.dt
        obs_dict["obj_vel_lin"] = sim.data.object_velocity(self.obj_bid, "body")[0]
        obs_dict["obj_velr"] = sim.data.object_velocity(self.obj_bid, "body")[1]
        obs_dict["muscle_len"] = np.nan_to_num(sim.data.actuator_length.copy())
        obs_dict["muscle_vel"] = np.nan_to_num(sim.data.actuator_velocity.copy())
        obs_dict["muscle_force"] = np.nan_to_num(sim.data.actuator_force.copy())
        return obs_dict

    def get_obs_ids_dict(self):
        spec_dict = {key: [] for key in self.ALL_OBS_KEYS}
        # Define the specs of each observation component
        for joint_name in self.joint_names:
            spec_dict["hand_jnt"].append([joint_name, "position", "angular", "joint"])
            spec_dict["hand_jnt_vel"].append(
                [joint_name, "velocity", "angular", "joint"]
            )

        for muscle_name in self.muscle_names:
            spec_dict["muscle_len"].append([muscle_name, "position", "muscle"])
            spec_dict["muscle_vel"].append([muscle_name, "velocity", "muscle"])
            spec_dict["muscle_force"].append([muscle_name, "force", "muscle"])
            spec_dict["act"].append([muscle_name, "activation", "muscle"])

        for coord in ["x", "y", "z"]:
            spec_dict["obj_pos"].append(
                ["object", "pen", "id_1", "position", "linear", coord]
            )
            spec_dict["obj_des_pos"].append(
                ["target", "pen", "id_1", "position", "linear", coord]
            )
            spec_dict["obj_err_pos"].append(
                ["target", "pen", "id_1", "position", "error", "linear", coord]
            )
            spec_dict["obj_rot"].append(
                ["object", "pen", "id_1", "position", "angular", coord]
            )
            spec_dict["obj_des_rot"].append(
                ["target", "pen", "id_1", "position", "angular", coord]
            )
            spec_dict["obj_err_rot"].append(
                ["target", "pen", "id_1", "position", "error", "angular", coord]
            )
            spec_dict["obj_vel_lin"].append(
                ["object", "pen", "id_1", "velocity", "linear", coord]
            )
            spec_dict["obj_velr"].append(
                ["object", "pen", "id_1", "velocity", "angular", coord]
            )

        return spec_dict

    def get_action_ids(self):
        """Create a list of specifications per action component. E.g., for a muscle
        activation, [muscle_id, activation, muscle]
        """
        action_specs = [
            [VOCABULARY[muscle_name], VOCABULARY["activation"], VOCABULARY["muscle"]]
            for muscle_name in self.muscle_names
        ]
        return np.array(action_specs, dtype=np.float32)
