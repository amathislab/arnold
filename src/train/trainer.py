import json
import os
from dataclasses import dataclass, field
from typing import List

from stable_baselines3 import PPO
from models.ppo.ppo import MultiEnvPPO, PCGradMultiEnvPPO, SeparatedValueFinetunePPO
from train.algo_factory import AlgoFactory
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from algos.buffers import ExpertRolloutBuffer, ExpertDictRolloutBuffer
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer

@dataclass
class Trainer:
    algo: str
    envs: VecNormalize
    env_config_list: List[dict]
    load_model_path: str
    log_dir: str
    log_interval: int = 16
    model_config: dict = field(default_factory=dict)
    custom_logger: object = None
    callbacks: List[BaseCallback] = field(default_factory=list)
    old_vocabulary: dict = field(default_factory=dict)
    use_expert_actions: bool = False
    reset_std: bool = False

    def __post_init__(self):
        self.dump_configs(path=self.log_dir)
        self.agent: PPO = self._init_agent()

    def dump_configs(self, path: str) -> None:
        for config in self.env_config_list:
            env_name = config["env_name"]
            with open(
                os.path.join(path, f"{env_name}_config.json"), "w", encoding="utf8"
            ) as f:
                json.dump(config, f, indent=4, default=lambda _: "<not serializable>")
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf8") as f:
            json.dump(
                self.model_config, f, indent=4, default=lambda _: "<not serializable>"
            )

    def _init_agent(self):
        algo_class = AlgoFactory.get_algo_class(self.algo)
        if self.load_model_path is not None:
            print(f"Loading model from {self.load_model_path}")
            custom_objects = {"vocabulary": self.old_vocabulary}
            custom_objects.update(self.model_config)
            return algo_class.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=self.log_dir,
                custom_objects=custom_objects,
                reset_std=self.reset_std
            )
        print("\nNo model path provided. Initializing new model.\n")
        agent = algo_class(
            env=self.envs,
            verbose=2,
            tensorboard_log=self.log_dir,
            use_expert_actions=self.use_expert_actions,
            **self.model_config,
        )
        if self.custom_logger :
            agent.set_logger(self.custom_logger)

        return agent

    def train(self, total_timesteps: int) -> None:
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
            log_interval=self.log_interval
        )

    def save(self) -> None:
        self.agent.save(os.path.join(self.log_dir, "final_model.pkl"))
        self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))


@dataclass
class SingleEnvTrainer:
    algo: str
    envs: VecNormalize
    env_config: dict
    load_model_path: str
    log_dir: str
    model_config: dict = field(default_factory=dict)
    callbacks: List[BaseCallback] = field(default_factory=list)
    timesteps: int = 10_000_000
    log_interval: int = 16
    def __post_init__(self):
        self.dump_configs(path=self.log_dir)
        self.agent = self._init_agent()

    def dump_configs(self, path: str) -> None:
        with open(os.path.join(path, "env_config.json"), "w", encoding="utf8") as f:
            json.dump(
                self.env_config, f, indent=4, default=lambda _: "<not serializable>"
            )
        with open(os.path.join(path, "model_config.json"), "w", encoding="utf8") as f:
            json.dump(
                self.model_config, f, indent=4, default=lambda _: "<not serializable>"
            )

    def _init_agent(self):
        algo_class = AlgoFactory.get_algo_class(self.algo)
        if self.load_model_path is not None:
            algo = algo_class.load(
                self.load_model_path,
                env=self.envs,
                tensorboard_log=self.log_dir,
                custom_objects=self.model_config,
                device=self.model_config.get("device", "cuda"),
            )
            # Replace rollout buffer if the model is trained with BC
            if isinstance(algo.rollout_buffer, ExpertRolloutBuffer) and self.algo == "ppo":
                algo.rollout_buffer = RolloutBuffer(
                    buffer_size=algo.rollout_buffer.buffer_size,
                    observation_space=algo.rollout_buffer.observation_space,
                    action_space=algo.rollout_buffer.action_space,
                    device=algo.rollout_buffer.device,
                    gae_lambda=algo.rollout_buffer.gae_lambda,
                    gamma=algo.rollout_buffer.gamma,
                    n_envs=algo.rollout_buffer.n_envs,
                )
            elif isinstance(algo.rollout_buffer, ExpertDictRolloutBuffer) and self.algo == "ppo":
                algo.rollout_buffer = DictRolloutBuffer(
                    buffer_size=algo.rollout_buffer.buffer_size,
                    observation_space=algo.rollout_buffer.observation_space,
                    action_space=algo.rollout_buffer.action_space,
                    device=algo.rollout_buffer.device,
                    gae_lambda=algo.rollout_buffer.gae_lambda,
                    gamma=algo.rollout_buffer.gamma,
                    n_envs=algo.rollout_buffer.n_envs,
                )
            elif isinstance(algo.rollout_buffer, RolloutBuffer) and self.algo == "bc_ppo" :
                 algo.rollout_buffer = ExpertRolloutBuffer(
                    buffer_size=algo.rollout_buffer.buffer_size,
                    observation_space=algo.rollout_buffer.observation_space,
                    action_space=algo.rollout_buffer.action_space,
                    device=algo.rollout_buffer.device,
                    gae_lambda=algo.rollout_buffer.gae_lambda,
                    gamma=algo.rollout_buffer.gamma,
                    n_envs=algo.rollout_buffer.n_envs,
                )
            elif isinstance(algo.rollout_buffer, DictRolloutBuffer) and self.algo == "bc_ppo" :
                 algo.rollout_buffer = ExpertRolloutBuffer(
                    buffer_size=algo.rollout_buffer.buffer_size,
                    observation_space=algo.rollout_buffer.observation_space,
                    action_space=algo.rollout_buffer.action_space,
                    device=algo.rollout_buffer.device,
                    gae_lambda=algo.rollout_buffer.gae_lambda,
                    gamma=algo.rollout_buffer.gamma,
                    n_envs=algo.rollout_buffer.n_envs,
                )
            return algo
        print("\nNo model path provided. Initializing new model.\n")
        return algo_class(
            env=self.envs,
            verbose=2,
            tensorboard_log=self.log_dir,
            **self.model_config,
        )

    def train(self) -> None:
        self.agent.learn(
            total_timesteps=self.timesteps,
            callback=self.callbacks,
            reset_num_timesteps=False,
            log_interval=self.log_interval,
        )

    def save(self) -> None:
        self.agent.save(os.path.join(self.log_dir, "final_model.pkl"))
        self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))


if __name__ == "__main__":
    print("This is a module. Run main.py to train the agent.")
