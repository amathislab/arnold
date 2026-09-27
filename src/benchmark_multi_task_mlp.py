"""Benchmark MT-SAC and MT-PPO using the saved training task order."""

import os
import json
import argparse

import numpy as np
import tqdm

from stable_baselines3 import SAC
from models.ppo.ppo import MultiEnvPPO
from definitions import ROOT_DIR, ENV_CONFIG_PATH
from envs.utilities import create_vec_env
from evaluation import find_vecnormalize, summarize_episodes
from stable_baselines3.common.utils import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(
        prog="BenchmarkMultiTaskMLP",
        description="Benchmark a multi-task MLP policy (MT-SAC / MT-PPO) trained "
        "with src/main_sac_multi_task.py",
    )
    parser.add_argument("--load", type=str, required=True, help="Path to the model .zip")
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        choices=["sac", "ppo"],
        help="Algorithm used to train the policy. If omitted, read from args.json.",
    )
    parser.add_argument(
        "--vecnormalize",
        type=str,
        default=None,
        help="Path to the vecnormalize .pkl. If omitted, it is derived from --load.",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        default=None,
        help="Subset of training tasks to report. The full training task list is "
        "always instantiated (to keep env_id / normalization correct); this flag "
        "only filters which tasks are evaluated and saved. Default: all tasks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Max steps counted per episode. The env still runs to its natural "
        "termination; this only caps the metrics window (matches benchmark.py).",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save benchmark results to JSON file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/benchmarks/student_policies",
        help="Directory to save benchmark results",
    )
    return parser.parse_args()


def get_train_args(load_path):
    """Read the args.json saved next to the model at training time."""
    policy_dir = os.path.dirname(load_path)
    args_path = os.path.join(policy_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(
            f"Could not find args.json in {policy_dir}. It is required to recover "
            "the training task order, algorithm and num_memory_steps."
        )
    with open(args_path, "r") as f:
        return json.load(f)


def build_env_config_list(tasks, num_memory_steps, dense_reward):
    """Reproduce the env_config_list built in main_sac_multi_task.py."""
    env_config_list = []
    for task in tasks:
        task_cfg_name = f"arnold_{task}_config.json"
        if dense_reward:
            task_cfg_name = "dense_" + task_cfg_name
        env_config_path = os.path.join(ENV_CONFIG_PATH, task_cfg_name)
        with open(env_config_path, "r") as f:
            env_config = json.load(f)
        env_config["num_memory_steps"] = num_memory_steps
        env_config_list.append(env_config)
    return env_config_list


def save_results(scores, args):
    if not args.save_results:
        return
    out_dir = os.path.join(ROOT_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    policy_dir = os.path.dirname(args.load)
    policy_name = os.path.basename(policy_dir)
    checkpoint_name = os.path.basename(args.load).replace(".zip", "")
    run_name = f"{policy_name}_{checkpoint_name}"

    results_file = os.path.join(out_dir, f"{run_name}_results.json")
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved results to {results_file}")


def main():
    args = parse_args()
    set_random_seed(args.seed)

    train_args = get_train_args(args.load)
    tasks = train_args.get("tasks")
    if not tasks:
        raise ValueError("args.json does not contain a non-empty 'tasks' list.")
    # SAC runs may store algo as null in args.json; default to "sac".
    algo = args.algo or train_args.get("algo") or "sac"
    num_memory_steps = train_args.get("num_memory_steps", 5)
    dense_reward = train_args.get("dense_reward", False)

    # Tasks to actually report. The full training list is always instantiated so
    # that env_id (= index in the training order) and the per-signature
    # normalization stay identical to training time.
    report_tasks = set(args.task) if args.task else set(tasks)
    unknown = report_tasks - set(tasks)
    if unknown:
        raise ValueError(
            f"--task entries {sorted(unknown)} were not in the training task list "
            f"{tasks}."
        )

    print(f"Training task order (env_id): {list(enumerate(tasks))}")
    print(f"Algorithm: {algo}, num_memory_steps: {num_memory_steps}, "
          f"dense_reward: {dense_reward}")

    # ---- Load the policy ----
    if algo == "sac":
        model = SAC.load(args.load, device=args.device, buffer_size=1)
    else:
        model = MultiEnvPPO.load(args.load, device=args.device)
    model.policy.eval()

    # ---- Rebuild the exact training env stack ----
    env_config_list = build_env_config_list(tasks, num_memory_steps, dense_reward)
    vecnormalize_path = args.vecnormalize or find_vecnormalize(args.load)
    if not os.path.isfile(vecnormalize_path):
        raise FileNotFoundError(vecnormalize_path)
    print(f"Loading vecnormalize from {vecnormalize_path}")

    venv = create_vec_env(
        env_config_list=env_config_list,
        num_envs_per_config=1,
        load_env_path=vecnormalize_path,
        multi_env=True,
        old_vocabulary=None,
        seed=args.seed,
    )
    # Freeze normalization statistics for evaluation.
    venv.training = False
    venv.norm_reward = False

    num_envs = venv.num_envs
    assert num_envs == len(tasks), (
        f"Expected one sub-env per task ({len(tasks)}), got {num_envs}."
    )
    max_episode_steps = venv.env_method("get_episode_horizon")
    set_random_seed(args.seed)
    venv.seed(args.seed)

    # Per-env episode target: 0 for tasks we are not reporting (they still step
    # along with the rest of the batch, we just ignore their results).
    targets = np.array(
        [args.num_episodes if t in report_tasks else 0 for t in tasks], dtype=int
    )

    # ---- Roll out ----
    episodes = [
        {"cum_rewards": [], "steps": [], "solved_counts": []} for _ in range(num_envs)
    ]
    done_counts = np.zeros(num_envs, dtype=int)
    counting = np.ones(num_envs, dtype=bool)  # decouple metric window from env episode
    cur_cum = np.zeros(num_envs)
    cur_steps = np.zeros(num_envs, dtype=int)
    cur_solved = np.zeros(num_envs)

    obs = venv.reset()
    total_target = int(targets.sum())
    with tqdm.tqdm(total=total_target, desc="Evaluating multi-task") as pbar:
        while np.any(done_counts < targets):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, dones, infos = venv.step(action)
            raw_rewards = venv.get_original_reward()

            for i in range(num_envs):
                if counting[i]:
                    cur_cum[i] += raw_rewards[i]
                    cur_steps[i] += 1
                    rwd_dict = infos[i].get("rwd_dict", {})
                    cur_solved[i] += float(rwd_dict.get("solved", 0.0))

                # Close the metrics window either at the natural episode end or
                # when the --num_steps cap is hit (env keeps running until done).
                cap_hit = (
                    args.num_steps is not None and cur_steps[i] >= args.num_steps
                )
                if counting[i] and (dones[i] or cap_hit):
                    if done_counts[i] < targets[i]:
                        episodes[i]["cum_rewards"].append(cur_cum[i])
                        episodes[i]["steps"].append(int(cur_steps[i]))
                        episodes[i]["solved_counts"].append(cur_solved[i])
                        done_counts[i] += 1
                        pbar.update(1)
                    counting[i] = False

                # Reset accumulators when the underlying env actually resets.
                if dones[i]:
                    cur_cum[i] = 0.0
                    cur_steps[i] = 0
                    cur_solved[i] = 0.0
                    counting[i] = True

    venv.close()

    # ---- Aggregate per task (merging duplicate tasks in the training list) ----
    scores = {}
    merged = {}
    for i, task in enumerate(tasks):
        if task not in report_tasks or done_counts[i] == 0:
            continue
        if task not in merged:
            merged[task] = {"cum_rewards": [], "steps": [], "solved_counts": [],
                            "mes": max_episode_steps[i]}
        merged[task]["cum_rewards"].extend(episodes[i]["cum_rewards"])
        merged[task]["steps"].extend(episodes[i]["steps"])
        merged[task]["solved_counts"].extend(episodes[i]["solved_counts"])

    for task, ep in merged.items():
        scores[task] = summarize_episodes(ep, ep["mes"])
        scores[task].update(seed=args.seed, deterministic=args.deterministic,
                            normalize_obs=bool(venv.norm_obs), num_steps=args.num_steps)

    print(json.dumps(scores, indent=2))
    save_results(scores, args)


if __name__ == "__main__":
    main()
