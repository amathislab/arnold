import numpy as np

import gymnasium as gym
import warnings
import mujoco
import time
from typing import Optional
import logging

from scipy.spatial.transform import Rotation as sRot

class BaseEnv(gym.Env):

    def __init__(self, cfg):
        print("BaseEnv: Original")
        self.clip_actions = False # "flag, used in setting the action space and when building pd actions scales"
        self.render_mode = "human" # "human or rgb array"

        self.headless = cfg.run.headless # "render a screen or not"
        self.sim_timestep_inv = cfg.env.sim_timestep_inv # "inverse of simulation timestep"
        self.sim_timestep = 1.0 / self.sim_timestep_inv # "simulation timestep"
        self.control_freq_inv = cfg.env.control_frequency_inv
        self.cur_t = 0 # Current simulation time step
        self.dt = self.sim_timestep * self.control_freq_inv # control time step
        self.fast_forward = cfg.run.fast_forward # "fast forward the simulation"
        # self.sim_timestep_inv / self.control_freq_inv should be 30.0

        # ... various rendering parameters
        self.viewer = None
        self.renderer = None
        self.camera = -1
        self.paused = False
        self.disable_reset = False
        self.follow = False
        self._render_camera = None

    def reset(self, seed: Optional[int] = None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): The random seed for the environment. Defaults to None.
            options (dict, optional): Additional options for resetting the environment; they are plugged into gym.Env.reset(). Defaults to None.

        Returns:
            tuple: A tuple containing the observation and info after the reset.
        """
        super().reset(seed=seed, options=options)
        self.cur_t = 0

        observation = self.compute_observations()

        if self.render_mode == "human":
            self.render()

        return observation, {}
    
    def compute_observations(self):
        raise NotImplementedError

    def compute_info(self):
        raise NotImplementedError
    
    def step(self, action: np.ndarray):
        """
        Takes a step in the environment.

        Args:
            action: The action to take in the environment. Must be compatible with the action space.

        Returns:
            observation: The current observation of the environment - cf compute_observations().
            reward: Scalar - the reward obtained from the environment. It is a weighted sum of several reward components.
            terminated: A boolean indicating whether the episode is terminated, i.e. a fail state has been reached.
            truncated: A boolean indicating whether the episode is truncated, i.e. the maximum number of steps has been reached.
            info: Additional information about the step - cf post_physics_step().
        """
        # apply actions
        self.pre_physics_step(action)

        # step physics and render each frame
        self.physics_step(action)

        # compute observations, rewards, resets, ...
        observation, reward, terminated, truncated, info = self.post_physics_step(action)

        # if human render update the visualizer
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def pre_physics_step(self, action):
        raise NotImplementedError
    
    def physics_step(self, action):
        raise NotImplementedError
    
    def post_physics_step(self, action):
        raise NotImplementedError
    
    def render(self, mode="human"):
        """
        Renders the environment.

        If the environment is not set to headless mode, it creates a viewer and updates the rendering based on the render mode.

        Returns:
            If render mode is "rgb_array", returns the rendered pixels as an array.
        """
        if not self.headless:
            if self.viewer is None and self.renderer is None:
                self.create_viewer()
            
            if self.render_mode == "human" or mode=="human":
                self.viewer.sync()
                if self.follow:
                    self.viewer.cam.lookat = self.mj_data.qpos[:3]
                if not self.fast_forward:
                    time.sleep(1. / 100)
            
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self._create_renderer()
            camera_to_use = (
                self._render_camera
                if self.camera == -1 and self._render_camera is not None
                else self.camera
            )
            self.renderer.update_scene(self.mj_data, camera=camera_to_use)
            pixels = self.renderer.render()
            return pixels

    def close(self):
        """
        Closes the environment viewer if it exists.
        """
        if self.viewer is not None:
            self.viewer.close()
    
    def seed(self, seed: Optional[int] = None):
        """
        Set the random seed for the environment.

        Args:
            seed (Optional[int]): The random seed to set. If None, a random seed will be used.
        """
        super().reset(seed=seed)

    def create_sim(self, xml_path: str):
        """
        Creates a simulation environment using the specified XML file.

        Args:
            xml_path (str): The path to the XML file.
        """
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_timestep

    def _create_renderer(self):
        self.renderer = mujoco.Renderer(
            self.mj_model,
            width=640,
            height=480,
        )  # MJ offline renderer
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data)
        self._render_camera = self._build_render_camera()

    def _build_render_camera(self):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.mj_model, camera)

        camera.lookat[:] = np.array([0.0, 0.0, 0.0])
        desired_camera_pos = np.array([-4.0, 0.0, 4.5])

        view_vec = camera.lookat - desired_camera_pos
        distance = np.linalg.norm(view_vec)
        if distance == 0.0:
            distance = 1.0
            view_vec = np.array([1.0, 0.0, 0.0])

        camera.distance = float(distance)
        camera.azimuth = float(np.degrees(np.arctan2(view_vec[1], view_vec[0])))
        ratio = np.clip(view_vec[2] / distance, -1.0, 1.0)
        camera.elevation = float(np.degrees(np.arcsin(ratio)))

        return camera

    def create_viewer(self):
        if not self.headless and self.render_mode == "human":
            print("human")
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self.key_callback)
            
        if not self.headless and self.render_mode == "rgb_array":
            self._create_renderer()

    def key_callback(self, keycode):
        print(keycode)
        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Paused {self.paused}")
        elif chr(keycode) == "R":
            self.reset()
        elif chr(keycode) == "M":
            self.disable_reset = not self.disable_reset
            print(f"Disable reset {self.disable_reset}")
        elif chr(keycode) == "F":
            self.follow = not self.follow
            print(f"Follow {self.follow}")
