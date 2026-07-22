import numpy as np
import collections
import gym
from myosuite.envs.myo.myochallenge.relocate_v0 import RelocateEnvV0
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler
from envs.env_mixins import SpecsObsMixin
from vocabulary import VOCABULARY


class CustomRelocateEnv(RelocateEnvV0):  # RelocateEnvV0Phase2
    CUSTOM_RWD_KEYS_AND_WEIGHTS = {
        "done": 0,
        "act_reg": 0,
        "sparse": 0,
        "solved": 1,
        "alive": 0,
        "pos_dist": 0,
        "rot_dist": 0,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        BaseV0.__init__(
            self,
            model_path=model_path,
            obsd_model_path=obsd_model_path,
            seed=seed,
            env_credits=self.MYO_CREDIT,
        )
        self._setup(**kwargs)

    def _setup(
        self,
        target_xyz_range,  # target position range (relative to initial pos)
        target_rxryrz_range,  # target rotation range (relative to initial rot)
        obs_keys: list = RelocateEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = CUSTOM_RWD_KEYS_AND_WEIGHTS,
        pos_th=0.025,  # position error threshold
        rot_th=0.262,  # rotation error threshold
        drop_th=0.50,  # drop height threshold
        lift_th=0.02,
        contact_th=0.005,
        reach_z_offset=0,
        pos_z_offset=0,
        obj_rel_target_pos=(0, 0, 0),
        **kwargs,
    ):
        self.palm_sid = self.sim.model.site_name2id("S_grasp")
        self.tip0 = self.sim.model.site_name2id("THtip")
        self.tip1 = self.sim.model.site_name2id("IFtip")
        self.tip2 = self.sim.model.site_name2id("MFtip")
        self.tip3 = self.sim.model.site_name2id("RFtip")
        self.tip4 = self.sim.model.site_name2id("LFtip")
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.target_xyz_range = target_xyz_range
        self.target_rxryrz_range = target_rxryrz_range
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th
        self.lift_th = lift_th
        self.contact_th = contact_th
        self.init_obj_z = 0
        self.reach_z_offset = reach_z_offset
        self.pos_z_offset = pos_z_offset
        self.obj_rel_target_pos = obj_rel_target_pos
        self.obj_shift_pos = self.sim.data.site_xpos[self.object_sid] + np.array(
            self.obj_rel_target_pos
        )

        super()._setup(
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            target_xyz_range=target_xyz_range,
            target_rxryrz_range=target_rxryrz_range,
            pos_th=pos_th,
            rot_th=rot_th,
            drop_th=drop_th,
            **kwargs,
        )
        keyFrame_id = 0 if self.obj_xyz_range is None else 1
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["hand_qpos"] = sim.data.qpos[:-7].copy()
        obs_dict["hand_qpos_corrected"] = sim.data.qpos[:-6].copy()
        obs_dict["hand_qvel"] = sim.data.qvel[:-6].copy() * self.dt
        obs_dict["obj_pos"] = sim.data.site_xpos[self.object_sid]
        obs_dict["goal_pos"] = sim.data.site_xpos[self.goal_sid]
        obs_dict["palm_pos"] = sim.data.site_xpos[self.palm_sid]
        obs_dict["palm_rot"] = mat2euler(
            np.reshape(sim.data.site_xmat[self.palm_sid], (3, 3))
        )
        obs_dict["tip0"] = sim.data.site_xpos[self.tip0]
        obs_dict["tip1"] = sim.data.site_xpos[self.tip1]
        obs_dict["tip2"] = sim.data.site_xpos[self.tip2]
        obs_dict["tip3"] = sim.data.site_xpos[self.tip3]
        obs_dict["tip4"] = sim.data.site_xpos[self.tip4]
        obs_dict["pos_err"] = obs_dict["goal_pos"] - obs_dict["obj_pos"]
        obs_dict["reach_err"] = obs_dict["palm_pos"] - obs_dict["obj_pos"]
        obs_dict["obj_rot"] = mat2euler(
            np.reshape(sim.data.site_xmat[self.object_sid], (3, 3))
        )
        obs_dict["goal_rot"] = mat2euler(
            np.reshape(sim.data.site_xmat[self.goal_sid], (3, 3))
        )
        obs_dict["rot_err"] = obs_dict["goal_rot"] - obs_dict["obj_rot"]

        if sim.model.na > 0:
            obs_dict["act"] = sim.data.act[:].copy()
        return obs_dict

    def get_reward_dict(self, obs_dict):
        # print(obs_dict['time'])
        # print(self.obs_dict['obj_pos'][0][0][2])
        reach_dist = np.abs(
            np.linalg.norm(
                self.obs_dict["reach_err"] + np.array([0.0, 0.0, self.reach_z_offset]),
                axis=-1,
            )
        )
        pos_dist_offset = np.abs(
            np.linalg.norm(
                self.obs_dict["pos_err"] + np.array([0.0, 0.0, self.pos_z_offset]),
                axis=-1,
            )
        )
        pos_dist = np.abs(np.linalg.norm(self.obs_dict["pos_err"], axis=-1))
        rot_dist = np.abs(np.linalg.norm(self.obs_dict["rot_err"], axis=-1))
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )
        drop = reach_dist > self.drop_th
        obj_z = self.obs_dict["obj_pos"][:, :, 2]
        pos_dist_obj_z = np.abs(obj_z - self.pos_z_offset)

        reach_dist_contact = np.abs(
            np.linalg.norm(
                self.obs_dict["palm_pos"] - self.obs_dict["obj_pos"], axis=-1
            )
        )
        rot_palm_obj = np.abs(
            np.linalg.norm(
                self.obs_dict["palm_rot"] - self.obs_dict["obj_rot"], axis=-1
            )
        )

        reach_dist_xy = np.abs(
            np.linalg.norm(self.obs_dict["reach_err"][:, :, :2], axis=-1)
        )
        reach_dist_z = np.abs(
            np.linalg.norm(
                self.obs_dict["reach_err"][:, :, 2] + self.reach_z_offset, axis=-1
            )
        )

        max_app = 0
        for ii in range(5):
            max_app += np.abs(
                np.linalg.norm(
                    obs_dict["tip" + str(ii)] - obs_dict["palm_pos"], axis=-1
                )
            )

        min_app = 0
        for ii in range(5):
            min_app += np.abs(
                np.linalg.norm(obs_dict["tip" + str(ii)] - obs_dict["obj_pos"], axis=-1)
            )

        close_bonus = 0

        obj_shift_reward = np.exp(
            -5 * np.linalg.norm(self.obs_dict["obj_pos"] - self.obj_shift_pos)
        )
        epsilon = 1e-4
        solved = [
            [
                (pos_dist < self.pos_th).item()
                and (rot_dist < self.rot_th).item()
                and (not drop)
            ]
        ]
        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist
                # Optional Keys
                # ('pos_dist', -1.*pos_mul*(pos_dist + np.log(pos_dist + epsilon**2))),
                ("pos_dist", np.exp(-5 * pos_dist_offset)),
                ("pos_dist_z", np.exp(-5 * pos_dist_obj_z)),
                ("rot_dist", -1.0 * (rot_dist + np.log(rot_dist + epsilon**2))),
                ("reach_dist", -1.0 * (reach_dist + np.log(reach_dist + epsilon**2))),
                (
                    "reach_dist_xy",
                    -1.0 * (reach_dist_xy + np.log(reach_dist + epsilon**2)),
                ),
                (
                    "reach_dist_z",
                    -1.0 * (reach_dist_z + np.log(reach_dist + epsilon**2)),
                ),
                ("alive", np.array([[not drop]])),
                ("lift_bonus", obj_z > self.init_obj_z + self.lift_th),
                ("max_app", 1.0 * max_app),
                ("min_app", -1.0 * min_app),
                ("contact_hand_obj", reach_dist_contact < self.contact_th),
                ("rot_palm_obj", -1.0 * rot_palm_obj),
                ("close_bonus", np.array([[1.0 * close_bonus]])),
                ("obj_shift", np.array([[obj_shift_reward]])),
                ("palm_dist", np.exp(-5 * reach_dist)),
                ("open_hand", -np.exp(-5 * max_app)),
                ("tip_dist", np.exp(-min_app)),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("sparse", -rot_dist - 10.0 * pos_dist),
                ("solved", np.array(solved)),
                (
                    "done",
                    np.squeeze(self.obs_dict["time"], axis=-1) > 1.5,
                ),  # (drop) and (self.obs_dict['time']>1)
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = (
            np.array([0, 2]) if rwd_dict["solved"] else np.array([2, 0])
        )
        self.sim.model.site_size[self.success_indicator_sid, :] = (
            np.array(
                [
                    0.25,
                ]
            )
            if rwd_dict["solved"]
            else np.array(
                [
                    0.1,
                ]
            )
        )
        return rwd_dict

    def step(self, action):
        if any(~np.isfinite(action)):
            print(action)
        obs, reward, done, info = super().step(action)
        obs = np.nan_to_num(obs)
        info.update(info.get("rwd_dict"))
        return obs, reward, done, info

    def reset(self, reset_qpos=None, reset_qvel=None):
        obs = super().reset(reset_qpos, reset_qvel)
        self.init_obj_z = np.array(self.sim.data.site_xpos[self.object_sid][2])
        # print(self.sim.data.site_xpos[self.object_sid])
        # print(self.sim.data.site_xpos[self.object_sid].shape)
        # print(self.init_obj_z)
        self.obj_shift_pos = self.sim.data.site_xpos[self.object_sid] + np.array(
            self.obj_rel_target_pos
        )
        return obs


class MuscleRelocateEnv(CustomRelocateEnv, SpecsObsMixin):
    ALL_OBS_KEYS = [
        "hand_qpos_corrected",
        "hand_qvel",
        "obj_pos",
        "goal_pos",
        "pos_err",
        "obj_rot",
        "goal_rot",
        "rot_err",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "act",
    ]

    OBS_KEYS = [
        "hand_qpos_corrected",
        "hand_qvel",
        "obj_pos",
        "goal_pos",
        "pos_err",
        "obj_rot",
        # "goal_rot",
        # "rot_err",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "act",
    ]

    RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 1,
        "solved": 2,
        "palm_dist": 0.1,
        "tip_dist": 0.1,
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
            model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, **kwargs
        )
        self.action_dim = self.sim.model.nu
        self.muscle_names = [
            self.sim.model.id2name(i, "actuator") for i in range(self.sim.model.na)
        ]
        self.joint_names = [
            self.sim.model.id2name(i, "joint") for i in range(self.sim.model.nq - 6)
        ] # -6 because of how the obs is specified in the challenge (after bugfix from org.)
        self._specs_obs_init_addon(
            include_adapt_state=include_adapt_state, num_memory_steps=num_memory_steps
        )
        self._init_done = True

    def _setup(
        self,
        target_xyz_range,  # target position range (relative to initial pos)
        target_rxryrz_range,  # target rotation range (relative to initial rot)
        obs_keys: list = OBS_KEYS,
        weighted_reward_keys: dict = RWD_KEYS_AND_WEIGHTS,
        pos_th=0.025,  # position error threshold
        rot_th=0.262,  # rotation error threshold
        drop_th=0.50,  # drop height threshold
        lift_th=0.02,
        contact_th=0.005,
        reach_z_offset=0,
        pos_z_offset=0,
        obj_rel_target_pos=(0, 0, 0),
        **kwargs,
    ):
        super()._setup(
            target_xyz_range=target_xyz_range,
            target_rxryrz_range=target_rxryrz_range,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            pos_th=pos_th,
            rot_th=rot_th,
            drop_th=drop_th,
            lift_th=lift_th,
            contact_th=contact_th,
            reach_z_offset=reach_z_offset,
            pos_z_offset=pos_z_offset,
            obj_rel_target_pos=obj_rel_target_pos,
            **kwargs,
        )

    def reset(self, reset_qpos=None, reset_qvel=None):
        obs = super().reset(reset_qpos, reset_qvel).astype(np.float32)
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
        obs_dict["muscle_len"] = np.nan_to_num(sim.data.actuator_length.copy())
        obs_dict["muscle_vel"] = np.nan_to_num(sim.data.actuator_velocity.copy())
        obs_dict["muscle_force"] = np.nan_to_num(sim.data.actuator_force.copy())
        return obs_dict

    def get_obs_ids_dict(self):
        spec_dict = {key: [] for key in self.ALL_OBS_KEYS}
        # Define the specs of each observation component
        for joint_name in self.joint_names:
            spec_dict["hand_qpos_corrected"].append([joint_name, "position", "angular", "joint"])
            spec_dict["hand_qvel"].append([joint_name, "velocity", "angular", "joint"])

        for coord in ["x", "y", "z"]:
            spec_dict["obj_pos"].append(["object", "id_1", "position", "linear", coord])
            spec_dict["goal_pos"].append(
                ["target", "object", "id_1", "position", "linear", coord]
            )
            spec_dict["pos_err"].append(
                ["target", "object", "id_1", "position", "error", "linear", coord]
            )
            spec_dict["obj_rot"].append(
                ["object", "id_1", "position", "angular", coord]
            )
            spec_dict["goal_rot"].append(
                ["target", "object", "id_1", "position", "angular", coord]
            )
            spec_dict["rot_err"].append(
                ["target", "object", "id_1", "position", "error", "angular", coord]
            )

        for muscle_name in self.muscle_names:
            spec_dict["muscle_len"].append([muscle_name, "position", "muscle"])
            spec_dict["muscle_vel"].append([muscle_name, "velocity", "muscle"])
            spec_dict["muscle_force"].append([muscle_name, "force", "muscle"])
            spec_dict["act"].append([muscle_name, "activation", "muscle"])
        
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
