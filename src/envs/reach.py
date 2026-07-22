import numpy as np
from myosuite.envs.myo.myobase.reach_v0 import ReachEnvV0
from envs.env_mixins import SpecsObsMixin
from vocabulary import VOCABULARY


class CleanReachEnv(ReachEnvV0):
    RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "solved": 1.0,
        "act_reg": 1.0,
    }
    def _setup(self,
            target_reach_range:dict,
            far_th = .35,
            obs_keys:list = ReachEnvV0.DEFAULT_OBS_KEYS,
            weighted_reward_keys:dict = RWD_KEYS_AND_WEIGHTS,
            **kwargs,
        ):
        return super()._setup(
            target_reach_range, far_th, obs_keys, weighted_reward_keys, **kwargs
        )

    def reset(self):
        obs = super().reset().astype(np.float32)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return obs.astype(np.float32), reward, done, info


class MuscleReachEnv(ReachEnvV0, SpecsObsMixin):
    ALL_OBS_KEYS = [
        "qpos",
        "qvel",
        "tip_pos",
        "target_pos",
        "reach_err",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "act",
    ]

    OBS_KEYS = [
        "qpos",
        "qvel",
        "tip_pos",
        "target_pos",
        "reach_err",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "act",
    ]

    RWD_KEYS_AND_WEIGHTS = {
        "reach": 1.0,
        "solved": 1.0,
        "act_reg": 1.0,
    }

    def __init__(
        self,
        model_path,
        obsd_model_path=None,
        seed=None,
        include_adapt_state=True,
        num_memory_steps=5,
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
            self.sim.model.id2name(i, "joint") for i in range(self.sim.model.nq)
        ]
        self.tip_names = [self.sim.model.id2name(i, "site") for i in self.tip_sids]
        self._specs_obs_init_addon(include_adapt_state, num_memory_steps)
        self._init_done = True

    def _setup(
        self,
        target_reach_range: dict,
        far_th=0.35,
        obs_keys: list = OBS_KEYS,
        weighted_reward_keys: dict = RWD_KEYS_AND_WEIGHTS,
        **kwargs,
    ):
        super()._setup(
            target_reach_range, far_th, obs_keys, weighted_reward_keys, **kwargs
        )

    def reset(self):
        obs = super().reset().astype(np.float32)
        obs = self.create_history_reset_obs(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # print(done)
        # if not hasattr(self, "total") :
        #     self.total = 0
        # else :
        #     self.total += 1
        # if self.total==5 :
        #     ForkedPdb().set_trace()
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
            spec_dict["qpos"].append([joint_name, "position", "angular", "joint"])
            spec_dict["qvel"].append([joint_name, "velocity", "angular", "joint"])

        for tip_name in self.tip_names:
            for coord in ["x", "y", "z"]:
                spec_dict["tip_pos"].append(
                    [tip_name, "position", "linear", coord, "tip"]
                )
                spec_dict["target_pos"].append(
                    [tip_name, "position", "target", "linear", coord, "tip"]
                )
                spec_dict["reach_err"].append(
                    [tip_name, "position", "error", "linear", coord, "tip"]
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
