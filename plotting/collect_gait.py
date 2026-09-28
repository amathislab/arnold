"""Collect deterministic Walk-to-point muscle state, contacts and root velocity."""
import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

from definitions import ENV_CONFIG_PATH
from envs.environment_factory import EnvironmentFactory
from envs.expert_wrapper import ExpertWrapper
from envs.np_transform_utils import calc_heading
from envs.utilities import create_vec_env
from analysis.rollouts import find_vecnormalize, load_policy, simulation_env
from models.ppo.helpers import call_muscle_transformer_policy


def place_goal(base, distance):
    heading = calc_heading(base.mj_data.body("root").xquat[None, :])[0]
    base.goal_pos[:2] = base.mj_data.qpos[:2] + distance * np.array([np.cos(heading), np.sin(heading)])
    base.goal_pos[2] = 0.94


def collect_episode(env, policy, normalizer, seed, goal_distance, num_steps=None, moving_goal=False):
    set_random_seed(seed)
    observation, _ = env.reset(seed=seed)
    base = simulation_env(env)
    place_goal(base, goal_distance)
    observation = base.create_history_reset_obs(base.get_obs().astype(np.float32))
    signals = {name: [] for name in ("activations", "foot_contacts", "root_vel")}
    done = False
    while not done:
        if moving_goal:
            place_goal(base, goal_distance)
            current = base.get_obs().astype(np.float32)
            base._obs_prev_list[-1] = current
            observation["obs"][:, -1] = current
        signals["activations"].append(base.mj_data.act.copy())
        signals["foot_contacts"].append(base.get_touch().copy())
        signals["root_vel"].append(base.proprioception["local_body_vel"][:3].copy())
        with torch.no_grad():
            if normalizer is None:
                action = env.get_expert_action(deterministic=True)
            else:
                observation = {key: value[None, ...] for key, value in observation.items()}
                observation = normalizer.normalize_single_obs_dict(observation, env_idx=0)
                action, _ = call_muscle_transformer_policy(policy, observation, deterministic=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
        observation, _, terminated, truncated, _ = env.step(np.asarray(action).flatten())
        done = terminated or truncated or (num_steps is not None and len(signals["activations"]) >= num_steps)
    return {name: np.asarray(values) for name, values in signals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--load", type=Path)
    source.add_argument("--expert", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_episodes", type=int, default=150)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--goal_distance", type=float, default=2.0)
    parser.add_argument("--moving_goal", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    set_random_seed(args.seed)
    config = json.loads((Path(ENV_CONFIG_PATH) / "arnold_kinesis_config.json").read_text())
    vocabulary, training_args = None, {}
    if args.load:
        vocabulary = json.loads((args.load.parent / "vocabulary.json").read_text())
        training_args = json.loads((args.load.parent / "args.json").read_text())
        config["num_memory_steps"] = training_args.get("num_memory_steps", config["num_memory_steps"])
    env = EnvironmentFactory.create(**config)
    normalizer = None
    if args.expert:
        env = ExpertWrapper(env, "kinesis", device=args.device)
        policy = env.expert_policy
    else:
        policy = load_policy(args.load, args.device, vocabulary)
        normalizer = create_vec_env(env_config_list=[config], multi_env=True,
            load_env_path=find_vecnormalize(str(args.load)), old_vocabulary=vocabulary, seed=args.seed)
        normalizer.training, normalizer.norm_reward = False, False
    policy.to(args.device)
    policy.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(args.output, "x") as file:
            file.attrs.update(seed=args.seed, deterministic=True, fs=1.0 / env.dt,
                              goal_distance=args.goal_distance, moving_goal=args.moving_goal, timing="before_step")
            model = simulation_env(env).mj_model
            names = [model.actuator(i).name for i in range(model.na)]
            file.create_dataset("muscle_names", data=np.asarray(names, dtype=h5py.string_dtype()))
            for episode in range(args.num_episodes):
                signals = collect_episode(env, policy, normalizer, args.seed + episode,
                                          args.goal_distance, args.num_steps, args.moving_goal)
                group = file.create_group(f"episode_{episode}")
                group.attrs["episode_length"] = len(signals["activations"])
                for name, values in signals.items():
                    group.create_dataset(name, data=values, compression="gzip", shuffle=True)
                print(f"Episode {episode}: {len(signals['activations'])} steps")
    finally:
        env.close()
        if normalizer is not None:
            normalizer.close()


if __name__ == "__main__":
    main()
