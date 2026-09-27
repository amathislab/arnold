"""Shared benchmark scoring and checkpoint normalization."""
from pathlib import Path
import numpy as np


def find_vecnormalize(model_path):
    path = Path(model_path)
    name = path.stem.replace("model", "model_vecnormalize") + ".pkl"
    candidates = [path.with_name(name)]
    if path.stem in ("model", "final_model"):
        candidates.append(path.with_name(path.stem.replace("model", "env") + ".pkl"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"No matching normalization file for {path}")


def get_episode_horizon(env):
    pending = [env]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        attributes = vars(current)
        horizon = attributes.get("_max_episode_steps")
        spec = attributes.get("spec")
        if horizon is None and spec is not None:
            horizon = spec.max_episode_steps
        if horizon is not None:
            return int(horizon)
        pending.extend(attributes[key] for key in ("env", "gym_env") if key in attributes)
    raise ValueError("The evaluation environment has no fixed episode horizon")


def summarize_episodes(episodes, horizon):
    rewards = np.asarray(episodes["cum_rewards"], dtype=float)
    lengths = np.asarray(episodes["steps"], dtype=int)
    solved = np.asarray(episodes["solved_counts"], dtype=float)
    step_rewards = rewards / lengths
    fractions = solved / horizon
    return {
        "avg_cum_reward": float(rewards.mean()),
        "std_cum_reward": float(rewards.std()),
        "avg_step_reward": float(step_rewards.mean()),
        "std_step_reward": float(step_rewards.std()),
        "avg_solved": float((solved > 0).mean()),
        "std_solved": float((solved > 0).std()),
        "avg_solved_steps": float(solved.mean()),
        "std_solved_steps": float(solved.std()),
        "avg_solved_step_frac": float(fractions.mean()),
        "avg_steps": float(lengths.mean()),
        "std_steps": float(lengths.std()),
        "max_episode_steps": int(horizon),
        "num_episodes": len(rewards),
        "episode_cum_rewards": rewards.tolist(),
        "episode_solve_step_fracs": fractions.tolist(),
        "episode_solved_steps": solved.tolist(),
        "episode_steps": lengths.tolist(),
    }
