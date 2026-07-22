import os
import joblib
import mujoco
import numpy as np
import sys
import gym
from envs.env_mixins import SpecsObsMixin
import envs.np_transform_utils as npt_utils
from typing import Tuple, Dict
from pathlib import Path
from collections import OrderedDict
from scipy.spatial.transform import Rotation as sRot
from myosuite.utils.quat_math import mat2euler
from vocabulary import VOCABULARY
from envs.myolegs_base_env import BaseEnv


path_root = Path(__file__).resolve().parents[2]
sys.path.append(str(path_root))

from envs.myolegs_task import MyoLegsTask

import logging
from definitions import ROOT_DIR

logger = logging.getLogger(__name__)


def compute_self_observations(
    body_pos: np.ndarray,
    body_rot: np.ndarray,
    body_vel: np.ndarray,
    body_ang_vel: np.ndarray,
) -> OrderedDict:
    """
    Computes observations of the agent's local body state relative to its root.

    Args:
        body_pos (np.ndarray): Global positions of the bodies.
        body_rot (np.ndarray): Global rotations of the bodies in quaternion format.
        body_vel (np.ndarray): Linear velocities of the bodies.
        body_ang_vel (np.ndarray): Angular velocities of the bodies.

    Returns:
        OrderedDict: Dictionary containing:
            - `root_h_obs`: Root height observation.
            - `local_body_pos`: Local body positions excluding root.
            - `local_body_rot_obs`: Local body rotations in tangent-normalized format.
            - `local_body_vel`: Local body velocities.
            - `local_body_ang_vel`: Local body angular velocities.
    """
    obs = OrderedDict()

    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:, 2:3]

    obs["root_h_obs"] = root_h

    heading_rot_inv_expand = heading_rot_inv[..., None, :]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1], axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],
        heading_rot_inv_expand.shape[2],
    )

    root_pos_expand = root_pos[..., None, :]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    obs["local_body_pos"] = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = npt_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = npt_utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    ###### Velocity ######
    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"] = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_body_ang_vel
    )
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    return obs


def compute_imitation_observations(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_vel: np.ndarray,
    time_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Computes imitation observations based on differences between current and reference states.

    Observations include local differences in body positions and velocities,
    as well as local reference positions relative to the root.

    Args:
        root_pos (np.ndarray): Root position of the current state.
        root_rot (np.ndarray): Root rotation of the current state.
        body_pos (np.ndarray): Current body positions.
        body_vel (np.ndarray): Current body velocities.
        ref_body_pos (np.ndarray): Reference body positions.
        ref_body_vel (np.ndarray): Reference body velocities.
        time_steps (int): Number of time steps for observation history.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - `diff_local_body_pos`: Differences in local body positions.
            - `diff_local_vel`: Differences in local body velocities.
            - `local_ref_body_pos`: Local reference body positions.
    """
    obs = OrderedDict()
    B, J, _ = body_pos.shape

    heading_inv_rot = npt_utils.calc_heading_quat_inv(root_rot)

    heading_inv_rot_expand = np.tile(
        heading_inv_rot[..., None, :, :].repeat(body_pos.shape[1], axis=1),
        (time_steps, 1, 1),
    )

    diff_global_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - body_pos.reshape(
        B, 1, J, 3
    )

    diff_local_body_pos_flat = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_body_pos.reshape(-1, 3)
    )

    obs["diff_local_body_pos"] = diff_local_body_pos_flat  # 1 * J * 3

    ##### Velocities
    diff_global_vel = ref_body_vel.reshape(B, time_steps, J, 3) - body_vel.reshape(
        B, 1, J, 3
    )
    obs["diff_local_vel"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3)
    )

    local_ref_body_pos = ref_body_pos.reshape(B, time_steps, J, 3) - root_pos.reshape(
        B, 1, 1, 3
    )  # preserves the body position
    obs["local_ref_body_pos"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3)
    )

    return obs


class KinesisEnv(BaseEnv):
    """
    Implements two high-level control tasks for the KINESIS(MyoLeg) framework directly inheriting from BaseEnv.
    Previous functionality from MyoLegsEnv and MyoLegsTask has been merged into this class.
    """

    REWARD_SPECS = {"w_energy": 0.1, "w_upright": 0.1, "k_energy": 0.05}
    PROPRIOCEPTIVE_INPUTS = [
        "root_height",
        "root_tilt",
        "local_body_pos",
        "local_body_rot",
        "local_body_vel",
        "local_body_ang_vel",
        "feet_contacts",
    ]

    MYOLEG_TRACKED_BODIES = [
        "root",
        "tibia_l",
        "tibia_r",
        "talus_l",
        "talus_r",
        "toes_l",
        "toes_r",
    ]

    TASK_INPUTS = ["diff_local_body_pos", "diff_local_vel", "local_ref_body_pos"]

    def __init__(
        self,
        reward_specs=REWARD_SPECS,
        kp_scale=1.0,
        kd_scale=1.0,
        headless=True,
        sim_timestep_inv=150,
        control_frequency_inv=5,
        fast_forward=True,
        xml_path=os.path.join(ROOT_DIR, "data/kinesis/xml/myolegs_onlylegs.xml"),
        proprioceptive_inputs=PROPRIOCEPTIVE_INPUTS,
        track_bodies=MYOLEG_TRACKED_BODIES,
        task_inputs=TASK_INPUTS,
        seed=None
    ):
        self.id = "Kinesis-v0"
        self.initial_pose = None
        self.global_offset = np.zeros([1, 3])
        self.gender_betas = [np.zeros(17)]  # current, all body shape is mean.

        # Store configuration
        self.reward_specs = reward_specs
        self._kp_scale = kp_scale
        self._kd_scale = kd_scale
        self.dtype = np.float32
        self.headless = headless  # "render a screen or not"
        self.render_mode = "human"  # "human or rgb array"
        self.sim_timestep_inv = sim_timestep_inv  # "inverse of simulation timestep"
        self.sim_timestep = 1.0 / self.sim_timestep_inv  # "simulation timestep"
        self.control_freq_inv = control_frequency_inv
        self.cur_t = 0  # Current simulation time step
        self.dt = self.sim_timestep * self.control_freq_inv  # control time step
        self.fast_forward = fast_forward  # "fast forward the simulation"
        self.viewer = None
        self.renderer = None
        self.camera = -1
        self.paused = False
        self.disable_reset = False
        self.follow = False
        self.proprioceptive_inputs = proprioceptive_inputs
        self.track_bodies = track_bodies
        self.task_inputs = task_inputs
        self.reward_info = {}
        self.goal_pos = np.zeros(3)
        self.previous_tracking_distance = None
        self._seed = seed

        # Create simulation and setup parameters
        self.create_sim(xml_path)
        self.setup_myolegs_params()

        # Setup spaces
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(self.get_obs_size()),
            np.inf * np.ones(self.get_obs_size()),
            dtype=self.dtype,
        )

        self.action_space = gym.spaces.Box(
            low=-np.ones(80),
            high=np.ones(80),
            dtype=self.dtype,
        )

    def get_obs_size(self):
        """
        Returns the total size of observations (proprioceptive + task).
        """
        return self.get_self_obs_size() + self.get_task_obs_size()

    def get_self_obs_size(self):
        """
        Returns the size of the proprioceptive observations.
        """
        inputs = self.proprioceptive_inputs
        tally = 0
        if "root_height" in inputs:
            tally += 1
        if "root_tilt" in inputs:
            tally += 4
        if "local_body_pos" in inputs:
            tally += 3 * self.num_bodies - 3
        if "local_body_rot" in inputs:
            tally += 6 * self.num_bodies
        if "local_body_vel" in inputs:
            tally += 3 * self.num_bodies
        if "local_body_ang_vel" in inputs:
            tally += 3 * self.num_bodies
        if "muscle_len" in inputs:
            tally += self.mj_model.nu
        if "muscle_vel" in inputs:
            tally += self.mj_model.nu
        if "muscle_force" in inputs:
            tally += self.mj_model.nu
        if "feet_contacts" in inputs:
            tally += 4

        return tally

    def setup_myolegs_params(self):
        """
        Sets up various parameters related to the MyoLeg environment.
        """
        self.mj_body_names = []
        for i in range(self.mj_model.nbody):
            body_name = self.mj_model.body(i).name
            self.mj_body_names.append(body_name)

        self.body_names = self.mj_body_names[1:]  # the first one is always world

        self.num_bodies = len(self.body_names)
        self.num_vel_limit = self.num_bodies * 3
        self.robot_body_idxes = [
            self.mj_body_names.index(name) for name in self.body_names
        ]
        self.robot_idx_start = self.robot_body_idxes[0]
        self.robot_idx_end = self.robot_body_idxes[-1] + 1

        self.qpos_lim = (
            np.max(self.mj_model.jnt_qposadr)
            + self.mj_model.jnt_qposadr[-1]
            - self.mj_model.jnt_qposadr[-2]
        )
        self.qvel_lim = (
            np.max(self.mj_model.jnt_dofadr)
            + self.mj_model.jnt_dofadr[-1]
            - self.mj_model.jnt_dofadr[-2]
        )

        # These are not required but are included for future reference
        geom_type_id = mujoco.mju_str2Type("geom")
        self.floor_idx = mujoco.mj_name2id(self.mj_model, geom_type_id, "floor")
        self.full_track_bodies = self.body_names
        self.reset_bodies = self.track_bodies
        self.track_bodies_id = [self.body_names.index(j) for j in self.track_bodies]
        self.reset_bodies_id = [self.body_names.index(j) for j in self.reset_bodies]

    def get_body_xpos(self):
        """Returns the body positions of the agent in X, Y, Z coordinates."""
        return self.mj_data.xpos.copy()[self.robot_idx_start : self.robot_idx_end]

    def get_body_xquat(self):
        """Returns the body rotations of the agent in quaternion"""
        return self.mj_data.xquat.copy()[self.robot_idx_start : self.robot_idx_end]

    def get_body_linear_vel(self):
        """Returns the linear velocity of the agent's body parts."""
        return (
            self.mj_data.sensordata[: self.num_vel_limit]
            .reshape(self.num_bodies, 3)
            .copy()
        )

    def get_body_angular_vel(self):
        """Returns the angular velocity of the agent's body parts."""
        return (
            self.mj_data.sensordata[self.num_vel_limit : 2 * self.num_vel_limit]
            .reshape(self.num_bodies, 3)
            .copy()
        )

    def get_touch(self):
        """Returns the touch sensor readings of the agent."""
        return self.mj_data.sensordata[self.num_vel_limit * 2 :].copy()

    def reset(self, seed=None, options=None):
        """Combines reset logic from MyoLegsTask and MyoLegsEnv"""
        # Reset task first
        self.reset_task(options)

        # Then reset the agent
        self.init_myolegs()
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Finally reset the base environment
        obs, _ =  super().reset(seed=seed, options=options)
        return obs

    def reset_task(self, options=None):
        """
        Resets the task to an initial state by reinitializing key motion parameters.

        The function selects a random start time for the reference motion that the agent
        uses to initialize its position.

        Args:
            options (dict, optional): Additional options for task resetting. Defaults to None.
        """
        self.goal_pos[:2] = np.random.uniform(-2, 2, size=2)
        self.goal_pos[2] = 0.94

    def compute_observations(self):
        """
        Computes full observations by combining proprioceptive and task observations.
        """
        prop_obs = self.compute_proprioception()
        task_obs = self.compute_task_obs()
        return np.concatenate([prop_obs, task_obs])

    def compute_tracking_distance(self) -> float:
        """
        Computes the Euclidean distance between the root position and the goal position.

        The distance is calculated in the 2D plane (ignoring the z-axis).

        Returns:
            float: The tracking distance.
        """
        body_pos = self.get_body_xpos()
        root_pos = body_pos[0]
        tracking_distance = np.linalg.norm(root_pos[:2] - self.goal_pos[:2])
        return tracking_distance

    def init_myolegs(self) -> None:
        """
        Initializes the MyoLegs environment.

        This method extends the superclass `init_myolegs` function, and
        additionally "warm starts" the task by computing the
        current tracking distance.
        """
        # Initialize motion states and poses
        self.initialize_motion_state()
        
        self.previous_tracking_distance = self.compute_tracking_distance()

    def initialize_motion_state(self) -> None:
        """
        Retrieves motion data from the motion library and sets the initial pose.
        """
        self.mj_data.qpos[:] = 0
        self.mj_data.qvel[:] = 0
        self.mj_data.qpos[2] = 0.94
        self.mj_data.qpos[3:7] = np.array([0.5, 0.5, 0.5, 0.5])

        initial_rot = sRot.from_euler("XYZ", [np.pi / 2, 0, -np.pi / 2])
        ref_qpos = np.array(
            [
                0.70731837,
                0.6811051,
                0.9361768,
                0.50487715,
                0.4650055,
                -0.50120455,
                -0.5269373,
            ]
        ).astype(np.float32)
        self.mj_data.qpos[:3] = ref_qpos[:3]
        self.mj_data.qpos[3:7] = (initial_rot * sRot.from_quat(ref_qpos[3:7])).as_quat()

        self.initial_pose = np.array(
            [
                0.57133061,
                -1.21943974,
                0.93766457,
                0.84073663,
                0.02309985,
                -0.02534906,
                0.54035704,
                0.05718979,
                -0.08046306,
                -0.36137755,
                0.00613998,
                0.00156276,
                0.34577042,
                0.03181017,
                0.15059691,
                0.23338945,
                0.22419988,
                0.03759186,
                -0.01772005,
                0.01923135,
                -0.11780726,
                0.07734922,
                -0.03445803,
                -0.33847641,
                -0.00639306,
                0.00154999,
                0.35934485,
                0.0327628,
                -0.16855011,
                0.20909414,
                0.2539858,
                0.03759526,
                -0.01772008,
                0.01923106,
                -0.11780739,
            ]
        ).astype(np.float32)
        self.mj_data.qpos[7:] = self.initial_pose[7:]

        # Set up velocity
        self.mj_data.qvel[:6] = 0
        self.mj_data.qvel[6:] = 0

        # Run kinematics
        mujoco.mj_kinematics(self.mj_model, self.mj_data)

    def compute_reset(self) -> Tuple[bool, bool]:
        """
        Determines whether the task should be reset based on termination and truncation conditions.

        Termination means failure, while truncation means success.

        Returns:
            tuple:
                - terminated (bool): If the agent falls before the episode times out, or if the episode times out but the agent is not close enough to the goal.
                - truncated (bool): If the agent is standing close enough to the goal when the episode times out.
        """
        fall_terminated = self.proprioception["root_height"] < 0.7
        timetout_terminated = self.cur_t >= 150
        truncated = (
            timetout_terminated
            and np.linalg.norm(self.get_body_xpos()[0, :2] - self.goal_pos[:2]) < 0.1
        )
        terminated = fall_terminated or (timetout_terminated and not truncated)

        return terminated, truncated

    def compute_task_obs(self) -> np.ndarray:
        """
        Computes task-specific observations used for imitation learning or control.

        This function calculates observations based on the current state of the body and
        its relation to the goal position. It includes positional and velocity differences,
        local reference positions, and biomechanical data (if enabled).

        The observations are structured to be useful for downstream tasks such as
        imitation learning or reinforcement learning.

        Returns:
            np.ndarray: A concatenated array of task observations, including positional
            differences, velocity differences, and local reference positions.

        Notes:
            - The reason we overload this function from `MyoLegsIm` is to set every
            reference position except the root to the current position of the agent.
            - If biomechanics recording is enabled, updates the list of foot contact states
            and joint positions for analysis.
            - The returned observations are computed in a normalized and relative format
            to ensure consistent scale and alignment.

        Observation Structure:
            - `diff_local_body_pos`: Difference in local body positions.
            - `diff_local_vel`: Difference in local body velocities.
            - `local_ref_body_pos`: Local reference body positions.
        """
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        root_rot = body_rot[:, 0]
        root_pos = body_pos[:, 0]

        body_pos_subset = body_pos[..., self.track_bodies_id, :]

        ref_pos_subset = body_pos_subset
        ref_pos_subset[..., 0, :] = self.goal_pos

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset = body_vel[..., self.track_bodies_id, :]

        zeroed_task_obs = compute_imitation_observations(
            root_pos,
            root_rot,
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            body_vel_subset,
            time_steps=1,
        )

        task_obs = {}
        task_obs["diff_local_body_pos"] = zeroed_task_obs["diff_local_body_pos"]
        task_obs["diff_local_vel"] = zeroed_task_obs["diff_local_vel"]
        task_obs["local_ref_body_pos"] = zeroed_task_obs["local_ref_body_pos"]

        return np.concatenate(
            [v.ravel() for v in task_obs.values()], axis=0, dtype=self.dtype
        )

    def compute_reward(self, action):
        """
        Computes the reward for the current timestep based on task performance metrics.

        The reward is calculated as a weighted combination of several components (see below).

        Args:
            action (np.ndarray): The action applied at the current timestep.

        Returns:
            float: The computed reward for the current timestep.

        Reward Components:
            - `tracking_reward`: Proportional to the improvement in distance to the goal, scaled by a factor of 20.
            - `energy_reward`: Average power usage (negative reward for excessive energy consumption).
            - `upright_reward`: Encourages an upright posture based on orientation.
            - `solved`: Fixed bonus (100) for reaching the goal within a threshold distance (0.1 units).

        Updates:
            - `self.previous_tracking_distance`: Stores the current tracking distance for use in the next step.
            - `self.curr_power_usage`: Clears the current power usage data.
            - `self.reward_info`: Dictionary storing individual reward components for analysis.

        Notes:
            - The final reward is a weighted combination of components, with weights specified in `self.reward_specs`.
        """
        body_pos = self.get_body_xpos()
        root_pos = body_pos[0]

        current_tracking_distance = np.linalg.norm(root_pos[:2] - self.goal_pos[:2])
        tracking_reward = (
            self.previous_tracking_distance - current_tracking_distance
        ) * 20

        self.previous_tracking_distance = current_tracking_distance

        energy_reward = np.mean(self.curr_power_usage)
        self.curr_power_usage = []

        solved = 1 if current_tracking_distance < 0.1 else 0

        reward = (
            tracking_reward
            * (1 - self.reward_specs["w_energy"] - self.reward_specs["w_upright"])
            + energy_reward * self.reward_specs["w_energy"]
            + self.compute_upright_reward() * self.reward_specs["w_upright"]
            + solved
        )

        self.reward_info = {
            "tracking_reward": tracking_reward,
            "upright_reward": self.compute_upright_reward(),
            "energy_reward": energy_reward,
            "solved": solved,
        }

        return reward

    def compute_upright_reward(self) -> float:
        """
        Computes the reward for maintaining an upright posture.

        The reward is based on the angles of tilt in the forward and sideways directions,
        calculated using trigonometric components of the root tilt.

        Returns:
            float: The upright reward, where a value close to 1 indicates a nearly upright posture.
        """
        upright_trigs = self.proprioception["root_tilt"]
        fall_forward = np.angle(
            np.cos(upright_trigs[0]) + 1j * np.sin(upright_trigs[1])
        )
        fall_sideways = np.angle(
            np.cos(upright_trigs[2]) + 1j * np.sin(upright_trigs[3])
        )
        upright_reward = np.exp(-3 * (fall_forward**2 + fall_sideways**2))
        return upright_reward

    def compute_energy_reward(self, action: np.ndarray) -> float:
        """
        Computes the energy efficiency reward based on the L1 and L2 norms of the action.

        The reward penalizes high energy usage, with an exponential scaling defined
        by a configurable parameter.

        Args:
            action (np.ndarray): The action vector applied at the current step.

        Returns:
            float: The energy reward, where higher values indicate more efficient energy usage.
        """
        l1_energy = np.abs(action).sum()
        l2_energy = np.linalg.norm(action)
        energy_reward = -l1_energy - l2_energy
        energy_reward = np.exp(self.reward_specs["k_energy"] * energy_reward)
        return energy_reward

    def post_physics_step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Processes the environment state after each physics step.

        Increments the simulation time, computes observations, reward, and checks
        for termination or truncation conditions. Collects and returns additional
        information about the reward components.

        Args:
            action (np.ndarray): The action applied at the current step.

        Returns:
            Tuple:
                - obs (np.ndarray): Current observations.
                - reward (float): Reward for the current step.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information, including raw reward components.
        """
        if not self.paused:
            self.cur_t += 1
        obs = self.compute_observations()
        reward = self.compute_reward(action)
        terminated, truncated = self.compute_reset()
        if self.disable_reset:
            terminated, truncated = False, False
        info = {}
        info["rwd_dict"] = self.reward_info
        info["rwd_dict"]["alive"] = not (terminated or truncated)
        info.update(
            {f"{self.id}/{key}": val for key, val in self.reward_info.items()}
        )

        return obs, reward, terminated, truncated, info

    def step(self, myosuite_action):
        """
        Executes a single step in the environment with the given action.

        Args:
            action: The action to apply at the current step.

        Returns:
            Tuple:
                - observation (np.ndarray): Current observations after the step.
                - reward (float): Reward for the applied action.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information about the step, including reward details.
        """
        action = 1.0 / (1.0 + np.exp(-5.0 * (myosuite_action - 0.5)))
        self.physics_step(action)
        observation, reward, terminated, truncated, info = self.post_physics_step(
            action
        )

        return observation, reward, terminated or truncated, info

    def physics_step(self, action: np.ndarray = None) -> None:
        """
        Executes a physics step in the simulation with the given action.

        Depending on the control mode, computes muscle activations and applies them
        to the simulation. Tracks power usage during the step.

        Args:
            action (np.ndarray): The action to apply. If None, a random action is sampled.
        """
        self.curr_power_usage = []

        if action is None:
            action = self.action_space.sample()

        for i in range(self.control_freq_inv):
            if not self.paused:
                self.mj_data.ctrl[:] = action
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.curr_power_usage.append(self.compute_energy_reward(action))

    def get_task_obs_size(self) -> int:
        """
        Calculates the size of the task observation vector based on configured inputs.

        This function sums up the dimensions of the observation components specified
        in `self.cfg.run.task_inputs`. Each component contributes to the observation size
        based on the number of tracked bodies and its dimensionality (e.g., 3 for position or velocity).

        Returns:
            int: The total size of the task observation vector.
        """
        obs_size = 0
        if "diff_local_body_pos" in self.task_inputs:
            obs_size += 3 * len(self.track_bodies)
        if "diff_local_vel" in self.task_inputs:
            obs_size += 3 * len(self.track_bodies)
        if "local_ref_body_pos" in self.task_inputs:
            obs_size += 3 * len(self.track_bodies)

        return obs_size

    def compute_proprioception(self) -> np.ndarray:
        """
        Computes proprioceptive observations for the current simulation state.

        Updates the humanoid's body and actuator states, and generates observations
        based on the configured inputs.

        Returns:
            np.ndarray: Flattened array of proprioceptive observations.

        Notes:
            - The observations are also stored in the `self.proprioception` attribute.
        """
        mujoco.mj_kinematics(
            self.mj_model, self.mj_data
        )  # update xpos to the latest simulation values

        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        body_vel = self.get_body_linear_vel()[None,]
        body_ang_vel = self.get_body_angular_vel()[None,]

        obs_dict = compute_self_observations(body_pos, body_rot, body_vel, body_ang_vel)

        root_rot = sRot.from_quat(self.mj_data.qpos[3:7])
        root_rot_euler = root_rot.as_euler("xyz")

        myolegs_obs = OrderedDict()

        inputs = self.proprioceptive_inputs

        if "root_height" in inputs:
            myolegs_obs["root_height"] = obs_dict["root_h_obs"]  # 1
        if "root_tilt" in inputs:
            myolegs_obs["root_tilt"] = np.array(
                [
                    np.cos(root_rot_euler[1]),
                    np.sin(root_rot_euler[1]),
                    np.cos(root_rot_euler[2]),
                    np.sin(root_rot_euler[2]),
                ]
            )  # 4
        if "local_body_pos" in inputs:
            myolegs_obs["local_body_pos"] = obs_dict["local_body_pos"][
                0
            ]  # 3 * num_bodies
        if "local_body_rot" in inputs:
            myolegs_obs["local_body_rot"] = obs_dict["local_body_rot_obs"][
                0
            ]  # 6 * num_bodies
        if "local_body_vel" in inputs:
            myolegs_obs["local_body_vel"] = obs_dict["local_body_vel"][
                0
            ]  # 3 * num_bodies
        if "local_body_ang_vel" in inputs:
            myolegs_obs["local_body_ang_vel"] = obs_dict["local_body_ang_vel"][
                0
            ]  # 3 * num_bodies
        if "muscle_len" in inputs:
            myolegs_obs["muscle_len"] = np.nan_to_num(
                self.mj_data.actuator_length.copy()
            )  # num_actuators
        if "muscle_vel" in inputs:
            myolegs_obs["muscle_vel"] = np.nan_to_num(
                self.mj_data.actuator_velocity.copy()
            )  # num_actuators
        if "muscle_force" in inputs:
            myolegs_obs["muscle_force"] = np.nan_to_num(
                self.mj_data.actuator_force.copy()
            )  # num_actuators
        if "feet_contacts" in inputs:
            myolegs_obs["feet_contacts"] = self.get_touch()  # 4
        self.proprioception = myolegs_obs

        return np.concatenate(
            [v.ravel() for v in myolegs_obs.values()], axis=0, dtype=self.dtype
        )


class MuscleKinesisEnv(KinesisEnv, SpecsObsMixin):
    ALL_OBS_KEYS = [
        "qpos",
        "qvel",
        "feet_contact_force",
        "root_tilt",
        "muscle_length",
        "muscle_velocity",
        "muscle_force",
        "act",
        "root_vel",
        "target_error",
    ]
    OBS_KEYS = ALL_OBS_KEYS

    def __init__(
        self,
        include_adapt_state=False,
        num_memory_steps=30,
        obs_keys=OBS_KEYS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.id = "MuscleKinesis-v0"
        self.obs_keys = obs_keys
        self.action_dim = self.mj_model.na
        self.muscle_names = [
            self.mj_model.actuator(i).name for i in range(self.mj_model.na)
        ]
        self.joint_names = [
            self.mj_model.jnt(i).name for i in range(1, self.mj_model.njnt)
        ]  # Exclude root at first place
        self._specs_obs_init_addon(include_adapt_state, num_memory_steps)

    def reset(self, **kwargs):
        super().reset()
        obs = self.get_obs()
        obs = self.create_history_reset_obs(obs.astype(np.float32))
        return obs

    def step(self, action, **kwargs):
        _, reward, done, info = super().step(action)
        obs = self.get_obs()
        obs = self.create_history_obs(obs.astype(np.float32))
        return obs, reward, done, info

    def get_obs_dict(self):
        obs_dict = {}
        # proprioception
        obs_dict["qpos"] = self.mj_data.qpos[7:35].copy()
        obs_dict["qvel"] = self.mj_data.qvel[6:34].copy() * self.dt
        obs_dict["feet_contact_force"] = self.get_touch()  # heel and toes
        obs_dict["root_tilt"] = mat2euler(self.mj_data.body("root").xmat.reshape(3, 3))[
            :2
        ]

        obs_dict["muscle_length"] = np.nan_to_num(self.mj_data.actuator_length.copy())
        obs_dict["muscle_velocity"] = np.nan_to_num(
            self.mj_data.actuator_velocity.copy()
        )
        obs_dict["muscle_force"] = np.nan_to_num(self.mj_data.actuator_force.copy())

        obs_dict["act"] = self.mj_data.act[:].copy()

        # exteroception
        # obs_dict['model_root_pos'] = self.mj_data.qpos[:3].copy()  # remove
        # root_quat = self.mj_data.body("root").xquat.copy()

        # obs_dict['model_root_vel'] = self.mj_data.qvel[:3].copy()  # transform into egocentric
        self.compute_proprioception()
        obs_dict["root_vel"] = self.proprioception["local_body_vel"][:3]
        obs_dict["target_error"] = self.goal_pos - self.mj_data.qpos[:3]

        return obs_dict

    def get_obs_ids_dict(self):
        spec_dict = {key: [] for key in self.ALL_OBS_KEYS}

        for joint_name in self.joint_names:
            joint_name_split = joint_name.split("_")
            if "l" in joint_name_split:
                side_name = "left"
            elif "r" in joint_name_split:
                side_name = "right"
            else:
                raise ValueError(f"Unknown side {joint_name}")
            joint = "_".join(j for j in joint_name_split if j not in ["l", "r"])
            spec_dict["qpos"].append([joint, side_name, "position", "angular", "joint"])
            spec_dict["qvel"].append([joint, side_name, "velocity", "angular", "joint"])

        for muscle_name in self.muscle_names:
            muscle, side = muscle_name.split("_")
            if side == "l":
                side_name = "left"
            elif side == "r":
                side_name = "right"
            else:
                raise ValueError(f"Unknown side {side}")
            spec_dict["muscle_length"].append([muscle, side_name, "position", "muscle"])
            spec_dict["muscle_velocity"].append(
                [muscle, side_name, "velocity", "muscle"]
            )
            spec_dict["muscle_force"].append([muscle, side_name, "force", "muscle"])
            spec_dict["act"].append([muscle, side_name, "activation", "muscle"])

        for coord in ["x", "y"]:
            spec_dict["root_tilt"].append(["root", coord, "angular", "position"])

        spec_dict["feet_contact_force"] = [
            ["right", "foot", "contact", "force"],
            ["right", "toes", "contact", "force"],
            ["left", "foot", "contact", "force"],
            ["left", "toes", "contact", "force"],
        ]

        for coord in ["x", "y", "z"]:
            spec_dict["root_vel"].append(["root", coord, "linear", "velocity"])
            spec_dict["target_error"].append(
                ["root", coord, "position", "linear", "error"]
            )

        return spec_dict

    def get_action_ids(self):
        """Create a list of specifications per action component. E.g., for a muscle
        activation, [muscle_id, activation, muscle]
        """
        action_specs = []
        for muscle_name in self.muscle_names:
            muscle, side = muscle_name.split("_")
            if side == "l":
                side_name = "left"
            elif side == "r":
                side_name = "right"
            else:
                raise ValueError(f"Unknown side {side}")
            action_specs.append(
                [
                    VOCABULARY[muscle],
                    VOCABULARY[side_name],
                    VOCABULARY["activation"],
                    VOCABULARY["muscle"],
                ]
            )
        return np.array(action_specs, dtype=np.float32)

    def get_obs(self):
        """
        Get state based observations from the environemnt.
        Uses robot to get sensors, reconstructs the sim and recovers the sensors.
        """
        # get obs_dict using the observed information
        self.obs_dict = self.get_obs_dict()

        # recoved observation vector from the obs_dict
        _, obs = self.obsdict2obsvec(self.obs_dict, self.obs_keys)
        return obs

    # recover obsvec from obs_dict
    def obsdict2obsvec(self, obs_dict, ordered_obs_keys):
        # recover vec
        obsvec = np.zeros(0)
        for key in ordered_obs_keys:
            obsvec = np.concatenate(
                [obsvec, obs_dict[key].ravel()]
            )  # ravel helps with images
        # Keep the return values consistent with MyoSuite envs
        return None, obsvec
