import _srcpath  # noqa: F401
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import r2_score
from stable_baselines3.common.utils import set_random_seed

from analysis.subspaces import load_signals, project_actions, nmf_controls, performance_summary
from collect_signals import normalize_observation, policy_action
from definitions import ENV_CONFIG_PATH
from envs.environment_factory import EnvironmentFactory
from envs.utilities import create_vec_env
from analysis.rollouts import find_vecnormalize, get_episode_horizon, load_policy


def evaluate(env, policy, normalizer, projection, num_episodes, seed, task_index, num_steps):
    rows = []
    for episode in range(num_episodes):
        set_random_seed(seed + episode)
        obs, _ = env.reset(seed=seed + episode)
        states, done = None, False
        reward_sum, solved, length = 0., 0., 0
        while not done:
            normalized = normalize_observation(obs, normalizer, True, task_index)
            with torch.no_grad():
                action, _, states, _ = policy_action(policy, normalized, states, length == 0, False)
            obs, reward, terminated, truncated, info = env.step(projection(action))
            reward_sum += reward
            solved += float(info["rwd_dict"]["solved"])
            length += 1
            done = terminated or truncated or (num_steps is not None and length >= num_steps)
        rows.append(dict(episode=episode, seed=seed + episode, reward=reward_sum, solved_steps=solved, steps=length))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Arnold with PCA or NMF control projections")
    parser.add_argument("--load", type=Path, required=True)
    parser.add_argument("--method", choices=["pca", "nmf"], default="pca")
    parser.add_argument("--scope", choices=["task", "global"], default="task")
    parser.add_argument("--basis_policy", default="arnold")
    parser.add_argument("--signals", "--activations_dir", type=Path, required=True,
                        help="Compact task.h5 or legacy task_episode_N.h5 directory")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--n_fits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_dir", type=Path, default=Path("data/pca_analysis"))
    args = parser.parse_args()
    tasks = args.tasks or json.loads(Path("data/analysis/reproduction/signals.json").read_text())["tasks"]
    fits = args.n_fits if args.method == "nmf" else 1
    if fits <= 0 or args.num_episodes <= 0 or args.num_episodes % fits:
        parser.error("--num_episodes must be divisible by --n_fits")
    signal = "action_means" if args.method == "pca" else "muscle_controls"
    datasets = {task: load_signals(args.signals / (task + ".h5"), signal)[0] for task in tasks}
    combined = np.vstack(list(datasets.values())) if args.scope == "global" else None
    dimensions = args.dimensions or list(range(39 if args.method == "pca" else 38, 0, -1))
    maximum = 39 if args.method == "pca" else 38
    if any(dimension < 1 or dimension > maximum for dimension in dimensions):
        parser.error(f"--dimensions must be between 1 and {maximum}")
    training = json.loads((args.load.parent / "args.json").read_text())
    vocabulary = json.loads((args.load.parent / "vocabulary.json").read_text())
    policy = load_policy(args.load, args.device, vocabulary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves, outcomes = [], []
    shared_pca = PCA(n_components=39).fit(combined) if combined is not None and args.method == "pca" else None
    shared_nmf = {}
    for task in tasks:
        prefix = "dense_arnold_" if training.get("dense_reward", False) else "arnold_"
        config = json.loads((Path(ENV_CONFIG_PATH) / (prefix + task + "_config.json")).read_text())
        if "num_memory_steps" in training:
            config["num_memory_steps"] = training["num_memory_steps"]
        env = EnvironmentFactory.create(**config)
        normalizer = None
        try:
            normalizer = create_vec_env(env_config_list=[config], load_env_path=find_vecnormalize(str(args.load)),
                                        multi_env=True, old_vocabulary=vocabulary, seed=args.seed)
            normalizer.training, normalizer.norm_reward = False, False
            policy.observation_space = env.observation_space
            extractor = policy.features_extractor
            task_index = extractor.task_names.index(task) if getattr(extractor, "position_embedding", None) == "task_specific" else None
            dataset = datasets[task] if combined is None else combined
            pca = (shared_pca if shared_pca is not None else PCA(n_components=39).fit(dataset)) if args.method == "pca" else None
            horizon = get_episode_horizon(env)
            for dimension in dimensions:
                episodes, reconstruction = [], []
                for fit in range(fits):
                    if pca is not None:
                        components = pca.components_[:dimension]
                        projection = lambda action: project_actions(action, pca.mean_, components)
                        context = nullcontext()
                        covariance = np.cov(datasets[task].T)
                        reconstruction.append(np.trace(components @ covariance @ components.T) / np.trace(covariance))
                        np.savez_compressed(args.out_dir / f"{task}_{dimension}_{fit}.npz", components=components, mean=pca.mean_)
                    else:
                        if combined is not None and (dimension, fit) in shared_nmf:
                            nmf = shared_nmf[dimension, fit]
                        else:
                            nmf = NMF(n_components=dimension, init="nndsvdar", random_state=fit, max_iter=2000).fit(dataset)
                            if combined is not None:
                                shared_nmf[dimension, fit] = nmf
                        reconstruction.append(r2_score(datasets[task], nmf.inverse_transform(nmf.transform(datasets[task])), multioutput="variance_weighted"))
                        projection = lambda action: action
                        context = nmf_controls(env, nmf)
                        np.savez_compressed(args.out_dir / f"{task}_{dimension}_{fit}.npz", components=nmf.components_)
                    with context:
                        values = evaluate(env, policy, normalizer, projection, args.num_episodes // fits,
                                          args.seed + fit * (args.num_episodes // fits), task_index, args.num_steps)
                    for row in values:
                        row.update(method=args.method, scope=args.scope, basis_policy=args.basis_policy,
                                   task=task, dimension=dimension, fit=fit, horizon=horizon)
                    episodes.extend(values)
                summary = performance_summary(episodes, horizon)
                curves.append(dict(method=args.method, scope=args.scope, basis_policy=args.basis_policy,
                                   task=task, dimension=dimension, reconstruction=np.mean(reconstruction),
                                   n_fits=fits, **summary))
                outcomes.extend(episodes)
                pd.DataFrame(curves).to_csv(args.out_dir / "curves.csv", index=False)
                pd.DataFrame(outcomes).to_csv(args.out_dir / "episodes.csv", index=False)
                print(task, dimension, summary, flush=True)
        finally:
            env.close()
            if normalizer is not None:
                normalizer.close()


if __name__ == "__main__":
    main()
