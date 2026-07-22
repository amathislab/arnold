import os
import torch
import gymnasium as gym
import numpy as np
import json
from envs.loaders import load_expert_policy_and_env
from models.kinesis.policy_lattice import PolicyLattice
from definitions import ROOT_DIR
from gymnasium.core import Wrapper


class ExpertWrapper(Wrapper):
    def __init__(self, env, task_name, device="cuda", custom_expert_config_path=None):
        super().__init__(env)
        if custom_expert_config_path is not None:
            with open(custom_expert_config_path, "r") as f:
                custom_expert_config = json.load(f)
        else:
            custom_expert_config = {}
        if task_name == "kinesis" and "kinesis" not in custom_expert_config:
            self.using_kinesis_default_expert = True
            self.device = (
                torch.device("cuda") if device == "cuda" and torch.cuda.is_available() else torch.device("cpu")
            )
            self.expert_policy = self._load_kinesis_expert_policy()
            self.expert_policy.norm.training = (
                False  # Important: set normalization to eval mode
            )
        else:
            self.using_kinesis_default_expert = False
            exp_policy, exp_env, exp_norm, self.arnold_policy = load_expert_policy_and_env(
                task_name, device=device, custom_expert_config_dict=custom_expert_config
            )
            if self.arnold_policy:
                self.env_idx = exp_norm.env_ids_vec.index(
                    self.env.unwrapped.id
                )
            exp_norm.training = False
            self.task_name = task_name
            self.expert_policy = exp_policy
            self.expert_env = exp_env
            self.expert_vecnormalize = exp_norm
            self.episode_start = False
            self.lstm_states = None
        print(f"Env: {type(env)}, Adding Expert Actions")

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.using_kinesis_default_expert:
            if self.arnold_policy:
                self.expert_env.clear_history()
            else:
                self.expert_env.reset()
        self.episode_start = True
        self.lstm_states = None
        return obs

    def step(self, action):
        self.episode_start = False
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def get_expert_action(self, deterministic=True):
        if self.using_kinesis_default_expert:
            obs = self.env.unwrapped.compute_observations()
            with torch.no_grad():
                action = self.expert_policy.select_action(
                    torch.from_numpy(obs).to(torch.float32).to(self.device),
                    self.env.unwrapped.mj_data,
                    self.env.unwrapped.mj_model,
                    deterministic,
                )
        else:
            _, obs_vec = self.expert_env.unwrapped.obsdict2obsvec(
                self.unwrapped.obs_dict, self.expert_env.unwrapped.obs_keys
            )
            obs_vec = obs_vec.astype(np.float32)
            if self.arnold_policy: # Need to check if the expert is a transformer
                obs_dict = self.expert_env.create_history_obs(obs_vec)
                norm_obs = self.expert_vecnormalize.normalize_single_obs_dict(obs_dict, self.env_idx)
            else:
                norm_obs = self.expert_vecnormalize.normalize_obs(obs_vec)
            action, self.lstm_states = self.expert_policy.predict(
                norm_obs,
                self.lstm_states,
                episode_start=self.episode_start,
                deterministic=deterministic,
            )

        return action.flatten()

    def _load_kinesis_expert_policy(self):
        """Load the lattice policy expert"""
        kinesis_obs_space = gym.spaces.Box(
            -np.inf * np.ones(self.get_obs_size()),
            np.inf * np.ones(self.get_obs_size()),
            dtype=self.dtype,
        )
        policy = PolicyLattice(kinesis_obs_space, self.action_space)
        load_path = os.path.join(ROOT_DIR, "data/expert_policies/kinesis/model.pth")
        policy.load(load_path)
        return policy.to(self.device)
