import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import glob
import json
import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from definitions import ROOT_DIR

# Reuse the same mappings from plot_radar.py
TASK_NAME_MAPPING = {
    "hand_little_reach": "Little\n reach",
    "hand_index_reach": "Index\n reach",
    "hand_middle_reach": "Middle\n reach",
    "hand_ring_reach": "Ring\n reach",
    "hand_thumb_reach": "Thumb\n reach",
    "reorient": "Die\n reorient",
    "pen": "Pen\n reorient",
    "baoding_p1_cw": "Baoding\n CW",
    "baoding_p1_ccw": "Baoding\n CCW",
    "baoding_p2_overlap": "Baoding\n hard",
    "baoding_p2": "Baoding\n harder",
    "elbow_pose": "Elbow\n pose",
    "relocate": "Object\n relocation",
    "kinesis": "Walk to\n point",
}

TASK_METRIC_MAP = {
    "hand_little_reach": "solved_step_frac",
    "hand_index_reach": "solved_step_frac",
    "hand_middle_reach": "solved_step_frac",
    "hand_ring_reach": "solved_step_frac",
    "hand_thumb_reach": "solved_step_frac",
    "reorient": "solved_step_frac",
    "pen": "solved_step_frac",
    "baoding_p1_cw": "solved_step_frac",
    "baoding_p1_ccw": "solved_step_frac",
    "baoding_p2_overlap": "solved_step_frac",
    "baoding_p2": "solved_step_frac",
    "elbow_pose": "solved_step_frac",
    "relocate": "solved_step_frac",
    "kinesis": "solved_step_frac",
}


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def _collect_seed_results(method_dir):
    """Return the sorted per-seed *_results.json files under a final_benchmarks method dir."""
    pattern = os.path.join(
        ROOT_DIR, "data/final_benchmarks", method_dir, "*", "*_results.json"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results files found for method dir: {method_dir}")
    return files


def generate_latex_table(tasks, method_performances, task_name_mapping):
    """
    Generate a LaTeX table from the performance data.

    Args:
        tasks: List of task names
        method_performances: Dictionary mapping method names to lists of performances
        task_name_mapping: Dictionary mapping task keys to display names

    Returns:
        String containing LaTeX code for a table
    """
    # Start LaTeX table
    latex = "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\resizebox{\\textwidth}{!}{\n"
    latex += "\\begin{tabular}{l " + "c" * len(tasks) + " c}\n"
    latex += "\\toprule\n"

    # Header row with task names - using shortstack for proper line breaks within cells
    formatted_task_names = []
    for task in tasks:
        task_name = task_name_mapping[task]
        parts = task_name.split("\n")
        formatted_name = "\\shortstack{" + " \\\\ ".join(parts) + "}"
        formatted_task_names.append(formatted_name)

    latex += "Method & " + " & ".join(formatted_task_names) + " & Average \\\\\n"
    latex += "\\midrule\n"

    # Data rows
    for method, performances in method_performances.items():
        avg_performance = np.mean(performances)
        performance_strs = [f"{perf:.1f}" for perf in performances]
        latex += (
            f"{method} & "
            + " & ".join(performance_strs)
            + f" & {avg_performance:.1f} \\\\\n"
        )

    # End LaTeX table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "}\n"
    latex += "\\caption{Relative performance (\\%) of methods compared to single task experts}\n"
    latex += "\\label{tab:method_performance}\n"
    latex += "\\end{table}"

    return latex


def create_bar_plots_internal(subtract_baseline=False):
    """Generic bar plotting function with option to subtract baseline"""
    # Load results from data/final_benchmarks - one directory per method, with
    # one *_results.json per seed.
    method_dir_dict = {
        "BC": "bc",
        "OBC w/o obs norm": "obc_wo_obs_norm",
        "OBC": "obc",
        "OBC-PPO": "obc_ppo",
        "Arnold": "arnold",
    }

    color_dict = {
        "BC": "#CCB974",
        "OBC w/o obs norm": "#937860",
        "OBC": "#64B5CD",
        "OBC-PPO": "#55A868",
        "Arnold": "#4C72B0",
    }
    expert_dir = os.path.join(ROOT_DIR, "data/final_benchmarks/expert_policies")

    # Load results - a list of per-seed result dicts for each method.
    results_dict = {
        key: [load_results(f) for f in _collect_seed_results(method_dir)]
        for key, method_dir in method_dir_dict.items()
    }

    # Load expert results
    expert_results = {}
    for filename in os.listdir(expert_dir):
        if filename.endswith("_results.json"):
            results = load_results(os.path.join(expert_dir, filename))
            expert_results.update(results)

    # Get common tasks and sort them
    tasks = sorted(list(results_dict["Arnold"][0].keys()))

    # Store method performances for LaTeX table
    method_performances = {}
    method_average_performances = {}  # Store average performances

    # Prepare plot data
    n_tasks = len(tasks)
    n_methods = len(method_dir_dict)
    bar_width = 0.2  # Keeping the same bar width

    # Create figure with increased size
    plt.figure(figsize=(15, 5))  # Increased width from 12 to 15

    # Calculate positions for bars with increased spacing between groups
    group_spacing = 1.15  # Spacing factor between groups
    indices = np.arange(n_tasks) * group_spacing  # For task bars

    # Add the average bar with extra separation
    avg_separation = 0.4  # Increased separation before average bar
    avg_index = indices[-1] + group_spacing + avg_separation
    indices = np.append(indices, avg_index)  # Add average bar position to indices

    # Plot bars for each method
    for i, (method, results_list) in enumerate(results_dict.items()):
        performances = []
        performances_list = []
        sems = []  # Add list for SEMs
        for task in tasks:
            metric = TASK_METRIC_MAP[task]
            expert_score = expert_results[task][f"avg_{metric}"]

            # Mean relative performance across seeds
            seed_scores = [
                (results[task][f"avg_{metric}"] / expert_score) * 100
                for results in results_list
            ]
            relative_performance = float(np.mean(seed_scores))
            if subtract_baseline:
                relative_performance -= 100  # Subtract baseline for improvement plot
            performances.append(relative_performance)

            # Per-episode relative solved-step fractions, pooled across seeds
            episode_fracs = np.concatenate(
                [
                    np.asarray(results[task]["episode_solved_fracs"])
                    for results in results_list
                ]
            )
            performances_list.append(episode_fracs / expert_score * 100)

            # Calculate SEM
            # std = results[task][f"std_{metric}"] / expert_score * 100
            # n = results[task].get("n_episodes", 200)  # default to 50 if not specified
            # sem = std / np.sqrt(n)
            # sems.append(sem)

        # Calculate average performance and SEM
        avg_performance = np.mean(performances)
        avg_sem = np.std(performances) / np.sqrt(len(performances))

        # Store average performance
        method_average_performances[method] = avg_performance

        # Store performances for LaTeX table
        method_performances[method] = performances

        # Plot individual task bars - keep the same bar width but with adjusted positions
        offset = i * bar_width - (n_methods - 1) * bar_width / 2
        plt.bar(
            indices[:n_tasks] + offset,
            performances,
            bar_width,
            label=method,
            color=color_dict[method],
        )

        # Add scatter dots for individual data points
        for j, (task, performance, performance_list) in enumerate(zip(tasks, performances, performances_list)):
            # Add small random jitter to x position for better visibility
            x_pos = indices[j] + offset
            # Subsample up to 100 episodes evenly across the pooled seeds for a
            # representative scatter (avoids over-plotting all episodes).
            if len(performance_list) > 100:
                sample_idx = np.linspace(0, len(performance_list) - 1, 100).astype(int)
                y_positions = performance_list[sample_idx]
            else:
                y_positions = performance_list
            plt.scatter(x_pos * np.ones_like(y_positions), y_positions,
                       color=color_dict[method], alpha=0.15, s=20,
                       edgecolors=color_dict[method], linewidth=0.5)

        # Plot average bar with a pattern to distinguish it
        plt.bar(
            indices[n_tasks] + offset,  # Position at the end
            [avg_performance],
            bar_width,
            color=color_dict[method],
        )

        print(f"{method}: {avg_performance:.2f} ± {avg_sem:.2f}")

    # Add expert performance line
    baseline_value = 0 if subtract_baseline else 100
    baseline_label = "Expert baseline" if subtract_baseline else "Single task experts"
    plt.axhline(
        y=baseline_value, color="green", linestyle="--", alpha=0.5, label=baseline_label
    )

    # Customize plot
    y_label = "Performance improvement (%)" if subtract_baseline else "Performance (%)"
    plt.ylabel(y_label, fontsize=16)

    # Create x-tick labels with task names and "Average" at the end
    x_labels = [TASK_NAME_MAPPING[task] for task in tasks] + ["Average"]
    plt.xticks(indices, x_labels, rotation=45, ha="center", fontsize=14)

    # Add vertical line before average bar to visually separate it
    plt.axvline(
        x=indices[n_tasks - 1] + (avg_separation / 2) + (group_spacing / 2),
        color="gray",
        linestyle="--",
        alpha=0.5,
    )

    # Organize legend on four columns and place above the graph
    plt.legend(fontsize=16, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    plt.grid(True, axis="y", alpha=0.3)

    # Increase font size for y-axis tick labels (percentages)
    plt.yticks(fontsize=14)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Reduce whitespace on left and right sides
    plt.xlim(indices[0] - group_spacing / 1.5, indices[-1] + group_spacing / 1.5)

    # Save the plot with appropriate name
    suffix = "_improvement" if subtract_baseline else ""
    out_dir = os.path.join(ROOT_DIR, "data", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_basename = f"bar_plot_arnold_ablation{suffix}"
    out_path_png = os.path.join(out_dir, f"{out_basename}.png")
    out_path_svg = os.path.join(out_dir, f"{out_basename}.svg")
    plt.savefig(out_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_path_svg, dpi=300, bbox_inches="tight")
    print(f"Saved Arnold ablation bar plot to {out_path_png} and {out_path_svg}")
    plt.close()


def create_bar_plots():
    create_bar_plots_internal(subtract_baseline=False)


def create_improvement_bar_plots():
    """Create bar plots showing performance improvement relative to experts (subtracting 100%)"""
    # This function will reuse most of the logic from create_bar_plots() but plot performance - 100
    create_bar_plots_internal(subtract_baseline=True)


if __name__ == "__main__":
    create_bar_plots()
