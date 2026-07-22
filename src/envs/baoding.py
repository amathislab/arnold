import collections
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.utils.quat_math import mat2euler
from myosuite.envs.myo.myochallenge.baoding_v1 import BaodingEnvV1, Task
from envs.env_mixins import SpecsObsMixin
from vocabulary import VOCABULARY


class CleanBaodingEnv(BaodingEnvV1):
    def _setup(
        self,
        frame_skip: int = 10,
        drop_th=1.25,  # drop height threshold
        proximity_th=0.015,  # object-target proximity threshold
        goal_time_period=(5, 5),  # target rotation time period
        goal_xrange=(0.025, 0.025),  # target rotation: x radius (0.03)
        goal_yrange=(0.028, 0.028),  # target rotation: x radius (0.02 * 1.5 * 1.2)
        obj_size_range=(0.022, 0.022),  # Object size range. Nominal 0.022
        obj_mass_range=(0.043, 0.043),  # Object weight range. Nominal 43 gms
        obj_friction_change=(0, 0, 0),
        task_choice="ccw",  # cw / ccw / hold / random_dir / random
        obs_keys: list = BaodingEnvV1.DEFAULT_OBS_KEYS,
        weighted_reward_keys: list = BaodingEnvV1.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        limit_init_angle=np.pi,
        limit_sds_angle=0,
        initial_phase=0,
        **kwargs,
    ):
        self.task_choice = task_choice
        self.which_task = self.sample_task()
        self.drop_th = drop_th
        self.proximity_th = proximity_th
        self.goal_time_period = goal_time_period
        self.goal_xrange = goal_xrange
        self.goal_yrange = goal_yrange
        self.limit_init_angle = limit_init_angle
        self.initial_phase = initial_phase
        self.limit_sds_angle = limit_sds_angle

        # balls start at these angles
        #   1= yellow = top right
        #   2= pink = bottom left

        self.ball_1_starting_angle = 1.0 * np.pi / 4.0 + initial_phase
        self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        # init desired trajectory, for rotations
        self.center_pos = [-0.0125, -0.07]  # [-.0020, -.0522]
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        self.counter = 0
        self.goal = self.create_goal_trajectory(
            time_step=frame_skip * self.sim.model.opt.timestep, time_period=6
        )

        # init target and body sites
        self.object1_bid = self.sim.model.body_name2id("ball1")
        self.object2_bid = self.sim.model.body_name2id("ball2")
        self.object1_sid = self.sim.model.site_name2id("ball1_site")
        self.object2_sid = self.sim.model.site_name2id("ball2_site")
        self.object1_gid = self.sim.model.geom_name2id("ball1")
        self.object2_gid = self.sim.model.geom_name2id("ball2")
        self.target1_sid = self.sim.model.site_name2id("target1_site")
        self.target2_sid = self.sim.model.site_name2id("target2_site")
        self.sim.model.site_group[self.target1_sid] = 2
        self.sim.model.site_group[self.target2_sid] = 2

        # setup for task randomization
        self.obj_mass_range = {"low": obj_mass_range[0], "high": obj_mass_range[1]}
        self.obj_size_range = {"low": obj_size_range[0], "high": obj_size_range[1]}
        self.obj_friction_range = {
            "low": self.sim.model.geom_friction[self.object1_gid] - obj_friction_change,
            "high": self.sim.model.geom_friction[self.object1_gid]
            + obj_friction_change,
        }

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            frame_skip=frame_skip,
            **kwargs,
        )

        # reset position
        self.init_qpos[:-14] *= 0  # Use fully open as init pos
        self.init_qpos[0] = -1.57  # Palm up

    def get_reward_dict(self, obs_dict):
        # tracking error
        target1_dist = np.linalg.norm(obs_dict["target1_err"], axis=-1)
        target2_dist = np.linalg.norm(obs_dict["target2_err"], axis=-1)
        target_dist = target1_dist + target2_dist
        act_mag = (
            np.linalg.norm(self.obs_dict["act"], axis=-1) / self.sim.model.na
            if self.sim.model.na != 0
            else 0
        )

        # detect fall
        object1_pos = (
            obs_dict["object1_pos"][:, :, 2]
            if obs_dict["object1_pos"].ndim == 3
            else obs_dict["object1_pos"][2]
        )
        object2_pos = (
            obs_dict["object2_pos"][:, :, 2]
            if obs_dict["object2_pos"].ndim == 3
            else obs_dict["object2_pos"][2]
        )
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2)  # keep both balls up

        rwd_dict = collections.OrderedDict(
            (
                # Perform reward tuning here --
                # Update Optional Keys section below
                # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
                # Examples: Env comes pre-packaged with two keys pos_dist_1 and pos_dist_2
                # Optional Keys
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                # Must keys
                ("act_reg", -1.0 * act_mag),
                ("alive", ~is_fall),
                ("sparse", -target_dist),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    * (target2_dist < self.proximity_th)
                    * (~is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        # Sucess Indicator
        self.sim.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.sim.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict

    def reset(
        self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None
    ):
        # reset task
        self.which_task = self.sample_task()

        # reset counters
        self.counter = 0
        self.x_radius = self.np_random.uniform(
            low=self.goal_xrange[0], high=self.goal_xrange[1]
        )
        self.y_radius = self.np_random.uniform(
            low=self.goal_yrange[0], high=self.goal_yrange[1]
        )

        # reset goal
        if time_period == None:
            time_period = self.np_random.uniform(
                low=self.goal_time_period[0], high=self.goal_time_period[1]
            )
        self.goal = (
            self.create_goal_trajectory(time_step=self.dt, time_period=time_period)
            if reset_goal is None
            else reset_goal.copy()
        )

        # balls mass changes
        self.sim.model.body_mass[self.object1_bid] = self.np_random.uniform(
            **self.obj_mass_range
        )  # call to mj_setConst(m,d) is being ignored. Derive quantities wont be updated. Die is simple shape. So this is reasonable approximation.

        self.sim.model.body_mass[self.object2_bid] = self.sim.model.body_mass[
            self.object1_bid
        ]

        # balls friction changes
        self.sim.model.geom_friction[self.object1_gid] = self.np_random.uniform(
            **self.obj_friction_range
        )
        self.sim.model.geom_friction[self.object2_gid] = self.sim.model.geom_friction[
            self.object1_gid
        ]

        # balls size changes
        self.sim.model.geom_size[self.object1_gid] = self.np_random.uniform(
            **self.obj_size_range
        )
        self.sim.model.geom_size[self.object2_gid] = self.sim.model.geom_size[
            self.object1_gid
        ]

        # reset scene
        qpos = self.init_qpos.copy() if reset_pose is None else reset_pose
        qvel = self.init_qvel.copy() if reset_vel is None else reset_vel
        self.robot.reset(qpos, qvel)

        self.ball_1_starting_angle = 1.0 * np.pi / 4.0 + self.initial_phase
        self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        if self.limit_sds_angle > 0:
            # First set the random phase within the sds limit - it will move the targets at the desired angle
            # so that we can move the balls there by copying their position
            random_ball_phase = self.np_random.uniform(
                low=-self.limit_sds_angle, high=self.limit_sds_angle
            )
            self.ball_1_starting_angle = self.ball_1_starting_angle + random_ball_phase
            self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

            qpos = self.init_qpos.copy() if reset_pose is None else reset_pose
            qvel = self.init_qvel.copy() if reset_vel is None else reset_vel
            self.robot.reset(qpos, qvel)

            # Execute a step so that the targets move to the starting position
            self.step(-np.ones(39))

            # update ball positions
            obs_dict = self.get_obs_dict(self.sim)
            target1_pos = obs_dict["target1_pos"].copy()
            target2_pos = obs_dict["target2_pos"].copy()
            qpos[23] = target1_pos[0]  # ball 1 x-position
            qpos[24] = target1_pos[1]  # ball 1 y-position
            qpos[30] = target2_pos[0]  # ball 2 x-position
            qpos[31] = target2_pos[1]  # ball 2 y-position
            self.sim.set_state(qpos=qpos, qvel=qvel)
        else:
            qpos = self.init_qpos.copy() if reset_pose is None else reset_pose
            qvel = self.init_qvel.copy() if reset_vel is None else reset_vel
            self.robot.reset(qpos, qvel)

        random_target_phase = self.np_random.uniform(
            low=-self.limit_init_angle, high=self.limit_init_angle
        )
        self.ball_1_starting_angle = self.ball_1_starting_angle + random_target_phase
        self.ball_2_starting_angle = self.ball_1_starting_angle - np.pi

        return self.get_obs()

    def sample_task(self):
        if self.task_choice == "ccw":
            return Task(Task.BAODING_CCW)
        elif self.task_choice == "cw":
            return Task(Task.BAODING_CW)
        elif self.task_choice == "hold":
            return Task(Task.HOLD)
        elif self.task_choice == "random":
            return Task(self.np_random.choice(list(Task)))
        elif self.task_choice == "random_dir":
            return Task(self.np_random.choice([Task.BAODING_CW, Task.BAODING_CCW]))
        else:
            raise ValueError("Unknown task for baoding: ", self.task_choice)

    def get_obs(self):
        obs = super().get_obs().astype(np.float32)
        return obs

    def get_obs_dict(self, sim):
        obs_dict = super().get_obs_dict(sim)
        obs_dict["muscle_len"] = np.nan_to_num(sim.data.actuator_length.copy())
        obs_dict["muscle_vel"] = np.nan_to_num(sim.data.actuator_velocity.copy())
        obs_dict["muscle_force"] = np.nan_to_num(sim.data.actuator_force.copy())
        return obs_dict


class MuscleBaodingEnv(CleanBaodingEnv, SpecsObsMixin):
    ALL_OBS_KEYS = [
        "hand_pos",
        "hand_vel",
        "object1_pos",
        "object1_velp",
        "object2_pos",
        "object2_velp",
        "target1_pos",
        "target2_pos",
        "target1_err",
        "target2_err",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "object1_rot",
        "object2_rot",
        "object1_velr",
        "object2_velr",
        "act",
    ]

    OBS_KEYS = [
        "hand_pos",
        "hand_vel",
        "object1_pos",
        "object1_velp",
        "object2_pos",
        "object2_velp",
        "target1_pos",
        "target2_pos",
        "target1_err",
        "target2_err",
        "muscle_len",
        "muscle_vel",
        "muscle_force",
        "object1_rot",
        "object2_rot",
        "object1_velr",
        "object2_velr",
        "act",
    ]

    RWD_KEYS_AND_WEIGHTS = {
        "pos_dist_1": 1.0,
        "pos_dist_2": 1.0,
        "alive": 1.0,
        "solved": 5.0,
        "act_reg": 1.0
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
        self.muscle_names = [
            self.sim.model.id2name(i, "actuator") for i in range(self.sim.model.na)
        ]
        self.joint_names = [
            self.sim.model.id2name(i, "joint")
            for i in range(self.sim.model.nq - 14)  # 14 for ball's free joint
        ]
        self._specs_obs_init_addon(include_adapt_state, num_memory_steps)
        self._init_done = True

    def _setup(
        self,
        frame_skip: int = 10,
        drop_th=1.25,  # drop height threshold
        proximity_th=0.015,  # object-target proximity threshold
        goal_time_period=(5, 5),  # target rotation time period
        goal_xrange=(0.025, 0.025),  # target rotation: x radius (0.03)
        goal_yrange=(0.028, 0.028),  # target rotation: x radius (0.02 * 1.5 * 1.2)
        obj_size_range=(0.022, 0.022),  # Object size range. Nominal 0.022
        obj_mass_range=(0.043, 0.043),  # Object weight range. Nominal 43 gms
        obj_friction_change=(0, 0, 0),
        task_choice="ccw",  # cw / ccw / hold / random_dir / random
        obs_keys: list = OBS_KEYS,
        weighted_reward_keys: list = RWD_KEYS_AND_WEIGHTS,
        limit_init_angle=np.pi,
        limit_sds_angle=0,
        initial_phase=0,
        **kwargs,
    ):
        super()._setup(
            frame_skip=frame_skip,
            drop_th=drop_th,
            proximity_th=proximity_th,
            goal_time_period=goal_time_period,
            goal_xrange=goal_xrange,
            goal_yrange=goal_yrange,
            obj_size_range=obj_size_range,
            obj_mass_range=obj_mass_range,
            obj_friction_change=obj_friction_change,
            task_choice=task_choice,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            limit_init_angle=limit_init_angle,
            limit_sds_angle=limit_sds_angle,
            initial_phase=initial_phase,
            **kwargs,
        )

    def reset(self, reset_pose=None, reset_vel=None, reset_goal=None, time_period=None):
        obs = (
            super()
            .reset(
                reset_pose=reset_pose,
                reset_vel=reset_vel,
                reset_goal=reset_goal,
                time_period=time_period,
            )
            .astype(np.float32)
        )
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

        obs_dict["hand_vel"] = sim.data.qvel[
            :-12
        ].copy()  # 6*2 for ball's velocity components, rest should be hand
        obs_dict["muscle_len"] = np.nan_to_num(sim.data.actuator_length.copy())
        obs_dict["muscle_vel"] = np.nan_to_num(sim.data.actuator_velocity.copy())
        obs_dict["muscle_force"] = np.nan_to_num(sim.data.actuator_force.copy())

        obs_dict["object1_rot"] = mat2euler(
            np.reshape(sim.data.site_xmat[self.object1_sid], (3, 3))
        )
        obs_dict["object2_rot"] = mat2euler(
            np.reshape(sim.data.site_xmat[self.object2_sid], (3, 3))
        )
        obs_dict["object1_velr"] = sim.data.object_velocity(self.object1_bid, "body")[1]
        obs_dict["object2_velr"] = sim.data.object_velocity(self.object2_bid, "body")[1]

        return obs_dict

    def get_obs_ids_dict(self):
        spec_dict = {key: [] for key in self.ALL_OBS_KEYS}
        # Define the specs of each observation component
        for joint_name in self.joint_names:
            spec_dict["hand_pos"].append([joint_name, "position", "angular", "joint"])
            spec_dict["hand_vel"].append([joint_name, "velocity", "angular", "joint"])

        for muscle_name in self.muscle_names:
            spec_dict["muscle_len"].append([muscle_name, "position", "muscle"])
            spec_dict["muscle_vel"].append([muscle_name, "velocity", "muscle"])
            spec_dict["muscle_force"].append([muscle_name, "force", "muscle"])
            spec_dict["act"].append([muscle_name, "activation", "muscle"])

        for id in [1, 2]:
            for coord in ["x", "y", "z"]:
                spec_dict[f"object{id}_pos"].append(
                    ["object", "ball", f"id_{id}", "position", "linear", coord]
                )
                spec_dict[f"object{id}_velp"].append(
                    ["object", "ball", f"id_{id}", "velocity", "linear", coord]
                )
                spec_dict[f"target{id}_pos"].append(
                    ["target", "ball", f"id_{id}", "position", "linear", coord]
                )
                spec_dict[f"target{id}_err"].append(
                    ["target", "ball", f"id_{id}", "position", "error", "linear", coord]
                )
                spec_dict[f"object{id}_rot"].append(
                    ["object", "ball", f"id_{id}", "position", "angular", coord]
                )
                spec_dict[f"object{id}_velr"].append(
                    ["object", "ball", f"id_{id}", "velocity", "angular", coord]
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
