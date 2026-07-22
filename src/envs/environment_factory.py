import gymnasium as gym
# import pybullet_envs
import myosuite
from definitions import ENV_NAME_TO_ID


class EnvironmentFactory:
    """Static factory to instantiate and register gym environments by name."""

    @staticmethod
    def create(env_name, **kwargs):
        """Creates an environment given its name as a string, and forwards the kwargs
        to its __init__ function.

        Args:
            env_name (str): name of the environment

        Raises:
            ValueError: if the name of the environment is unknown

        Returns:
            gym.env: the selected environment
        """
        env_id = ENV_NAME_TO_ID.get(env_name)
        if env_id is None:
            print(
                "WARNING: environment name not recognized:",
                env_name,
                "Trying to create it with gym.make",
            )
            return gym.make(env_name, **kwargs)
        else:
            return gym.make("GymV21Environment-v0", env_id=env_id, make_kwargs=kwargs)
