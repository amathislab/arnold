import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from definitions import ROOT_DIR

# Reuse the same mappings from plot_radar.py
TASK_NAME_MAPPING = {
    "hand_little_reach": "Little reach",
    "hand_index_reach": "Index reach",
    "hand_middle_reach": "Middle reach",
    "hand_ring_reach": "Ring reach",
    "hand_thumb_reach": "Thumb reach",
    "reorient": "Die reorient",
    "pen": "Pen reorient",
    "baoding_p1_cw": "Baoding CW",
    "baoding_p1_ccw": "Baoding CCW",
    "baoding_p2_overlap": "Baoding hard",
    "baoding_p2": "Baoding harder",
    "elbow_pose": "Elbow pose",
    "relocate": "Object relocation",
    "kinesis": "Walk to point",
}

TASK_METRIC_MAP = {
    "hand_little_reach": "solved_steps",
    "hand_index_reach": "solved_steps",
    "hand_middle_reach": "solved_steps",
    "hand_ring_reach": "solved_steps",
    "hand_thumb_reach": "solved_steps",
    "reorient": "solved",
    "pen": "solved_steps",
    "baoding_p1_cw": "solved_steps",
    "baoding_p1_ccw": "solved_steps",
    "baoding_p2_overlap": "solved_steps",
    "baoding_p2": "solved_steps",
    "elbow_pose": "solved_steps",
    "relocate": "solved",
    "kinesis": "solved",
}


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def aggregate_seed_results(method_dir, base="data/final_benchmarks"):
    """Aggregate a multi-seed method dir (``<base>/<method_dir>/*/*_results.json``).

    Returns a single results dict (one entry per task) whose ``avg_*``/``std_*``
    fields are aggregated across seeds so it plugs into the shared plotting loop.
    The per-task ``std`` is the standard deviation of the per-seed means and
    ``n_episodes`` is the number of seeds, so the ``sem = std / sqrt(n_episodes)``
    formula below yields the SEM across seeds.
    """
    pattern = os.path.join(ROOT_DIR, base, method_dir, "*", "*_results.json")
    seed_files = sorted(glob.glob(pattern))
    if not seed_files:
        raise FileNotFoundError(f"No results files found: {pattern}")
    seed_results = [load_results(f) for f in seed_files]

    aggregated = {}
    for task in seed_results[0].keys():
        agg = {}
        for metric in ("solved_steps", "solved"):
            avg_key, std_key = f"avg_{metric}", f"std_{metric}"
            seed_means = [
                r[task][avg_key] for r in seed_results if avg_key in r[task]
            ]
            if seed_means:
                agg[avg_key] = float(np.mean(seed_means))
                agg[std_key] = float(np.std(seed_means))
        agg["n_episodes"] = len(seed_results)
        aggregated[task] = agg
    return aggregated


def load_single_results(subdir, base="data/final_benchmarks_extra"):
    """Load the single per-run results json under ``<base>/<subdir>/``."""
    pattern = os.path.join(ROOT_DIR, base, subdir, "*_results.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results files found: {pattern}")
    return load_results(files[0])


def create_bar_plots():
    # Load results. PPO, PPO w/o obs norm and MT-PPO come from the multi-seed
    # method dirs in data/final_benchmarks and are aggregated across seeds.
    # PPO w/o rew norm has no final_benchmarks equivalent, so its single run was
    # copied into data/final_benchmarks_extra/ppo_wo_rew_norm.
    results_dict = {
        "PPO": aggregate_seed_results("ppo_t_sv"),
        "PPO w/o rew norm": load_single_results("ppo_wo_rew_norm"),
        "PPO w/o obs norm": aggregate_seed_results("ppo_t"),
        "MT-PPO": aggregate_seed_results("mt_ppo"),
    }

    color_dict = {
        "PPO": "#C44E52",
        "PPO w/o rew norm": "#55A868",
        "PPO w/o obs norm": "#937860",
        "MT-PPO": "#7A68A8"
    }

    expert_dir = os.path.join(ROOT_DIR, "data/final_benchmarks/expert_policies")

    # Load expert results
    expert_results = {}
    for filename in os.listdir(expert_dir):
        if filename.endswith("_results.json"):
            results = load_results(os.path.join(expert_dir, filename))
            expert_results.update(results)

    # Get common tasks and sort them
    tasks = sorted(list(results_dict["PPO"].keys()))

    # Prepare plot data
    n_tasks = len(tasks)
    n_methods = len(results_dict)
    bar_width = 0.2

    # Create figure with increased size
    plt.figure(figsize=(9.5, 5))

    # Calculate positions for bars
    indices = np.arange(n_tasks)

    # Plot bars for each method
    for i, (method, results) in enumerate(results_dict.items()):
        performances = []
        sems = []  # Add list for SEMs
        for task in tasks:
            metric = TASK_METRIC_MAP[task]
            expert_score = expert_results[task][f"avg_{metric}"]
            task_score = results[task][f"avg_{metric}"]

            # Calculate relative performance
            relative_performance = (task_score / expert_score) * 100
            performances.append(relative_performance)

            # Calculate SEM
            std = results[task][f"std_{metric}"] / expert_score * 100
            n = results[task].get("n_episodes", 50)  # default to 50 if not specified
            sem = std / np.sqrt(n)
            sems.append(sem)

        offset = i * bar_width - (n_methods - 1) * bar_width / 2
        plt.bar(
            indices + offset,
            performances,
            bar_width,
            label=method,
            color=color_dict[method],
            yerr=sems,  # Add error bars
            capsize=3,  # Add caps to error bars
            error_kw={"elinewidth": 1},  # Make error bars more visible
        )
        print(
            f"{method}: {np.mean(performances):.2f} ± {(np.std(performances) / np.sqrt(len(performances))):.2f}"
        )

    # Add expert performance line
    plt.axhline(
        y=100, color="green", linestyle="--", alpha=0.5, label="Experts"
    )

    # Customize plot
    plt.xlabel("Tasks", fontsize=14)
    plt.ylabel("Performance (%)", fontsize=14)
    # plt.title("Multi-task Performance Comparison", fontsize=16)
    plt.xticks(
        indices, [TASK_NAME_MAPPING[task] for task in tasks], rotation=45, ha="right"
    )

    # Organize legend on four columns and place above the graph
    plt.legend(fontsize=12, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    plt.grid(True, axis="y", alpha=0.3)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    out_dir = os.path.join(ROOT_DIR, "data", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_basename = "bar_plot_ppo_ablation"
    out_path_png = os.path.join(out_dir, f"{out_basename}.png")
    out_path_svg = os.path.join(out_dir, f"{out_basename}.svg")
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_path_svg, dpi=300, bbox_inches="tight")
    print(f"Saved PPO ablation bar plot to {out_path_png} and {out_path_svg}")
    plt.close()


if __name__ == "__main__":
    create_bar_plots()
