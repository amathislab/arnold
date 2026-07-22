import numpy as np
from myosuite.envs.myo.myobase.pose_v0 import PoseEnvV0
from envs.env_mixins import SpecsObsMixin
from vocabulary import VOCABULARY


class CleanPoseEnv(PoseEnvV0):
    RWD_KEYS_AND_WEIGHTS = {
        "pose": 1.0,
        "solved": 1.0,
        "act_reg": 1.0,
    }
    def _setup(
        self,
        viz_site_targets: tuple = None,  # site to use for targets visualization []
        target_jnt_range: dict = None,  # joint ranges as tuples {name:(min, max)}_nq
        target_jnt_value: list = None,  # desired joint vector [des_qpos]_nq
        reset_type="init",  # none; init; random; sds
        target_type="generate",  # generate; switch; fixed
        obs_keys: list = PoseEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys: dict = RWD_KEYS_AND_WEIGHTS,
        pose_thd=0.35,
        weight_bodyname=None,
        weight_range=None,
        target_distance=1,  # for non-SDS curriculum, the target is set at a fraction of the full distance
        **kwargs,
    ):
        self.target_distance = target_distance
        super()._setup(
            viz_site_targets,
            target_jnt_range,
            target_jnt_value,
            reset_type,
            target_type,
            obs_keys,
            weighted_reward_keys,
            pose_thd,
            weight_bodyname,
            weight_range,
            **kwargs,
        )

    def reset(self, **kwargs):
        obs = super().reset().astype(np.float32)
        return obs

    def step(self, action, **kwargs):
        obs, reward, done, info = super().step(action)
        return obs.astype(np.float32), reward, done, info

    def get_target_pose(self):
        full_distance_target_pose = super().get_target_pose()
        init_pose = self.init_qpos.copy()
        target_pose = init_pose + self.target_distance * (
            full_distance_target_pose - init_pose
        )
        return target_pose


class MusclePoseEnv(PoseEnvV0, SpecsObsMixin):
    ALL_OBS_KEYS = [
        "qpos",
        "qvel",
        "act",
        "pose_err",
        "pose_target",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
    ]
    OBS_KEYS = [
        "qpos",
        "qvel",
        "act",
        "pose_err",
        "pose_target",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
    ]

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
        self.muscle_names = [
            self.sim.model.id2name(i, "actuator") for i in range(self.sim.model.na)
        ]
        self.joint_names = [
            self.sim.model.id2name(i, "joint") for i in range(self.sim.model.nq)
        ]
        self._specs_obs_init_addon(include_adapt_state, num_memory_steps)
        self._init_done = True

    def _setup(
        self,
        obs_keys: list = OBS_KEYS,
        **kwargs,
    ):
        super()._setup(
            obs_keys=obs_keys,
            **kwargs,
        )

    def reset(self, **kwargs):
        obs = super().reset().astype(np.float32)
        obs = self.create_history_reset_obs(obs)
        return obs

    def step(self, action, **kwargs):
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
        obs_dict["pose_target"] = self.target_jnt_value
        return obs_dict

    def get_obs_ids_dict(self):
        """For each observation component, we create a list of specifications. E.g., for
        a joint position, [joint_name, position, angular]
        """
        spec_dict = {key: [] for key in self.ALL_OBS_KEYS}
        # Define the specs of each observation component
        for joint_name in self.joint_names:
            spec_dict["qpos"].append([joint_name, "position", "angular", "joint"])
            spec_dict["qvel"].append([joint_name, "velocity", "angular", "joint"])
            spec_dict["pose_err"].append(
                [joint_name, "position", "angular", "error", "joint"]
            )
            spec_dict["pose_target"].append(
                [joint_name, "position", "angular", "target", "joint"]
            )

        for muscle_name in self.muscle_names:
            spec_dict["muscle_len"].append([muscle_name, "position", "muscle"])
            spec_dict["muscle_vel"].append([muscle_name, "velocity", "muscle"])
            spec_dict["muscle_force"].append([muscle_name, "force", "muscle"])
            spec_dict["act"].append([muscle_name, "activation", "muscle"])

        print("--------------------------------")
        for key, value in spec_dict.items():
            print(key, ":")
            print(value)
        print("--------------------------------")

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
