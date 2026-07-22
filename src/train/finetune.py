import os
import json
import torch
from dataclasses import dataclass, field
from typing import List
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from models.ppo.policies import MuscleTransformerPolicy
from train.algo_factory import AlgoFactory

@dataclass
class FinetuneTrainer() :

    algo: str
    policy_path: str
    model_config: dict
    envs: VecNormalize
    env_config_list: List[dict]
    learning_rate: float = 1e-5
    n_steps: int = 102400
    vf_coef: float = 0.5
    batch_size: int = 256
    log_interval: int = 1
    custom_logger: object = None
    device: str = "cuda"
    callbacks: List[BaseCallback] = field(default_factory=list)
    old_vocabulary: dict = field(default_factory=dict)

    def __post_init__(self):
        # self.dump_configs(path=self.log_dir)
        self.agent = self._init_agent()

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

        # Do lora/disable_grad here
        algo_class = AlgoFactory.get_algo_class(self.algo)
        agent: OnPolicyAlgorithm = algo_class(
            MuscleTransformerPolicy,
            env=self.envs,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            vf_coef=self.vf_coef,
            batch_size=self.batch_size,
            policy_kwargs=self.model_config,
        )
        if self.custom_logger :
            agent.set_logger(self.custom_logger)
        data_dict = torch.load(self.policy_path, map_location=self.device)
        agent.policy.load_state_dict(data_dict["state_dict"])
        return agent

    # def _fix_actor(self) :

    #     self.agent.policy.

    def train(self, total_timesteps: int) -> None:
        # self.agent.policy.features_extractor.requires_grad_(False)
        # self.agent.policy.mlp_extractor.requires_grad_(False)
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callbacks,
            log_interval=self.log_interval,
            reset_num_timesteps=False,
        )

    # def save(self) -> None:
    #     self.agent.save(os.path.join(self.log_dir, "final_model.pkl"))
    #     self.envs.save(os.path.join(self.log_dir, "final_env.pkl"))
