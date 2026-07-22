import gymnasium as gym
import numpy as np


class AnyMasses(gym.Env):
    """An arbitrary number of point masses can be controlled by the agent, which can apply forces on them.
    The goal is to reach a target positions with each mass. The observation is the position and velocity of each mass,
    and the action is the force applied to each mass. The reward is proportional to the negative distance
    to the target positions (made positive by an exponential filter).
    """

    def __init__(
        self,
        num_masses=2,
        mass_range=(0.1, 1),
        dt=0.01,
        max_force=1.0,
        init_radius=1.0,
        target_radius=1.0,
    ):
        self.num_masses = num_masses
        self.min_mass, self.max_mass = mass_range
        self.dt = dt
        self.max_force = max_force
        self.init_radius = init_radius
        self.target_radius = target_radius

        self.mass_positions = np.random.uniform(
            -self.max_position, self.max_position, (self.num_masses, 2)
        )
        self.mass_velocities = np.zeros((self.num_masses, 2))
        self.mass_forces = np
