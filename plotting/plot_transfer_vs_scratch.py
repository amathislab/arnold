import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from definitions import ROOT_DIR

# Reuse existing mappings and helper functions
from plot_student_policy_curves import (
    TASK_NAME_MAPPING,
    get_tb_data,
    running_average,
    load_expert_results,
    to_relative,
)


def list_matching_experiments(student_policies_dir, exp_number):
    """List all experiment directories that start with the given experiment number."""
    all_dirs = os.listdir(student_policies_dir)
    matching_dirs = [d for d in all_dirs if d.startswith(exp_number)]
    return matching_dirs


# Define experiment pairs and their tasks with correct task identifiers
EXPERIMENT_PAIRS = [
    ("180", "188", "pen"),  # Changed from "Pen reorient" to "pen"
    ("181", "189", "reorient"),  # Changed from "Die reorient" to "reorient"
    (
        "182",
        "190",
        "hand_middle_reach",
    ),  # Changed from "Middle reach" to "hand_middle_reach"
    (
        "183",
        "191",
        "hand_little_reach",
    ),  # Changed from "Little reach" to "hand_little_reach"
]


def main():
    # Setup paths
    student_policies_dir = os.path.join(
        ROOT_DIR, "data", "student_policies", "arnold_multi_task"
    )
    attribute = "solved"

    expert_results = load_expert_results()

    # Print available directories for each experiment number
    exp_numbers = ["180", "181", "182", "183", "188", "189", "190", "191"]
    for exp in exp_numbers:
        matches = list_matching_experiments(student_policies_dir, exp)
        if matches:
            print(f"\nExperiment {exp} matches:")
            for m in matches:
                print(f"  {m}")

    # Create 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()

    # Plot each pair of experiments
    for idx, (transfer_exp, scratch_exp, task) in enumerate(EXPERIMENT_PAIRS):
        print(
            f"Processing transfer experiment {transfer_exp} vs scratch experiment {scratch_exp}"
        )

        # Get data for transfer learning experiment
        transfer_dir = os.path.join(
            student_policies_dir, f"{transfer_exp}_arnold_{task}_bc_ppo_seed_0"
        )
        transfer_data = get_tb_data(transfer_dir, [attribute])

        # Get data for learning from scratch experiment
        scratch_dir = os.path.join(
            student_policies_dir, f"{scratch_exp}_arnold_{task}_bc_ppo_seed_0"
        )
        scratch_data = get_tb_data(scratch_dir, [attribute])

        ax = axs[idx]

        # Plot both curves
        for data, label, color in [
            (transfer_data, "Transfer", "#4C72B0"),
            (scratch_data, "From scratch", "#C44E52"),
        ]:
            if data:
                x_vals, y_vals = next(iter(data.values()))

                # Apply running average
                if len(y_vals) > 10:
                    window = 5
                    y_vals = running_average(y_vals, window)
                    x_vals = x_vals[window - 1 :]

                # Normalise to relative solved fraction (% of single-task expert)
                y_vals = to_relative(y_vals, task, expert_results)

                if x_vals[0] > 4e7:
                    x_vals = x_vals - x_vals[0]
                    y_vals[0] = 0
                else:
                    x_vals = np.insert(x_vals, 0, 0)
                    y_vals = np.insert(y_vals, 0, 0)

                ax.plot(x_vals, y_vals, label=label, color=color, linewidth=3.0)

        # Reference line at expert performance
        ax.axhline(100, color="grey", linestyle="--", linewidth=2.0, alpha=0.8, zorder=0)

        # Customize subplot
        readable_name = TASK_NAME_MAPPING.get(task, task)
        ax.set_title(readable_name, fontsize=24)
        ax.set_xlim([0, 2e6])
        ax.set_ylim([0, 120])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=20)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(20)

        # Add labels
        ax.set_xlabel("Steps", fontsize=22)
        ax.set_ylabel("Relative Performance (%)", fontsize=22)
        ax.legend(fontsize=20)

    # plt.suptitle("Transfer Learning vs Learning from Scratch", fontsize=28, y=1.02)
    plt.tight_layout()

    # Save the figure
    out_dir = os.path.join(ROOT_DIR, "data", "figures", "transfer_learning")
    os.makedirs(out_dir, exist_ok=True)
    out_basename = "transfer_vs_scratch_comparison"
    out_path_png = os.path.join(out_dir, f"{out_basename}.png")
    out_path_svg = os.path.join(out_dir, f"{out_basename}.svg")
    fig.savefig(out_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_path_svg, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_path_png} and {out_path_svg}")
    plt.close()


if __name__ == "__main__":
    main()
