import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from definitions import ROOT_DIR
from matplotlib.cm import plasma

# Add the task name mapping at the top of the file
TASK_NAME_MAPPING = {
    "hand_little_reach": "Little reach",
    "hand_index_reach": "Index reach",
    "hand_middle_reach": "Middle reach",
    "hand_ring_reach": "Ring reach",
    "hand_thumb_reach": "Thumb reach",
    "hand_pose": "Hand pose",
    "pen": "Pen reorient",
    "reorient": "Die reorient",
    "relocate": "Object relocation",
    "baoding_p1_cw": "Baoding CW",
    "baoding_p1_ccw": "Baoding CCW",
    "baoding_p2_overlap": "Baoding hard",
    "baoding_p2": "Baoding harder",
    "elbow_pose": "Elbow pose",
    "elbow_joint_pose": "Elbow joint",
    "finger_pose": "Finger pose",
    "kinesis": "Walk to point",
}

# The tensorboard "solved" scalar is a per-episode solved-step fraction in [0, 1],
# except for these tasks, which log it as a percentage in [0, 100].
PERCENT_SCALED_TASKS = {"kinesis"}

# Per-task x-axis limit. Most tasks converge well within DEFAULT_X_MAX; kinesis is
# trained for far longer and would otherwise be cut off mid-climb.
DEFAULT_X_MAX = 4e6
TASK_X_MAX = {"kinesis": 2e7}


def load_expert_results():
    """Load the single-task expert benchmark results, keyed by task name."""
    expert_dir = os.path.join(ROOT_DIR, "data", "final_benchmarks", "expert_policies")
    expert_results = {}
    for fname in os.listdir(expert_dir):
        if fname.endswith("_results.json"):
            with open(os.path.join(expert_dir, fname)) as f:
                expert_results.update(json.load(f))
    return expert_results


def to_relative(y_vals, task, expert_results):
    """Convert a tensorboard "solved" curve into relative performance (% of expert).

    Matches the normalisation used by plot_radar.py: the score is the ratio of the
    student's solved steps to the expert's avg_solved_steps. The tensorboard scalar
    logs the fraction of solved steps per episode, so it is first rescaled into
    solved steps by the task's episode length.
    """
    expert = expert_results.get(task)
    if expert is None:
        print(f"Warning: no expert results for '{task}', leaving curve unnormalised.")
        return y_vals
    expert_steps = expert["avg_solved_steps"]
    if not expert_steps:
        print(f"Warning: expert avg_solved_steps is 0 for '{task}'.")
        return y_vals
    solved_frac = y_vals / 100 if task in PERCENT_SCALED_TASKS else y_vals
    solved_steps = solved_frac * expert["max_episode_steps"]
    return (solved_steps / expert_steps) * 100


def get_data_from_tb_log(path, y, x="step", tb_config=None):
    if tb_config is None:
        tb_config = {}

    event_acc = EventAccumulator(path, tb_config)
    event_acc.Reload()

    if not isinstance(y, list):
        y = [y]

    out_dict = {}
    for attr_name in y:
        all_attr_names = event_acc.Tags()["scalars"]
        for full_attr_name in all_attr_names:
            if attr_name in full_attr_name:
                x_vals, y_vals = np.array(
                    [
                        (getattr(el, x), el.value)
                        for el in event_acc.Scalars(full_attr_name)
                    ]
                ).T
                out_dict[full_attr_name] = (x_vals, y_vals)

    return out_dict


def get_tb_data(experiment_dir, attributes):
    tb_dir_name = "PPO_0"  # The tensorboard directory name
    tb_dir_path = os.path.join(experiment_dir, tb_dir_name)

    if os.path.isdir(tb_dir_path):
        data_dict = {}
        folder_content = os.listdir(tb_dir_path)
        for tb_file_name in folder_content:
            tb_file_path = os.path.join(tb_dir_path, tb_file_name)
            data_dict_to_add = get_data_from_tb_log(tb_file_path, attributes)
            data_dict = extend_dict(data_dict, data_dict_to_add)
            # Sort timesteps and values
            for key in data_dict.keys():
                sort_idx = np.argsort(data_dict[key][0])
                data_dict[key] = (
                    data_dict[key][0][sort_idx],
                    data_dict[key][1][sort_idx],
                )
        return data_dict
    else:
        print("Warning: no tb dir at ", tb_dir_path)
        return None


def extend_dict(dict1, dict2):
    for key in dict2.keys():
        if key in dict1:
            dict1[key] = (
                np.concatenate((dict1[key][0], dict2[key][0])),
                np.concatenate((dict1[key][1], dict2[key][1])),
            )
        else:
            dict1[key] = dict2[key]
    return dict1


def running_average(x, window_size=100):
    """Simple running average filter.

    Args:
        x (array): Input signal
        window_size (int): Size of the averaging window
    """
    return np.convolve(x, np.ones(window_size) / window_size, mode="valid")


def main():
    # Setup paths and parameters
    student_policies_dir = os.path.join(
        ROOT_DIR, "data", "student_policies", "arnold_single_task"
    )
    attribute = "solved"  # The metric we want to plot

    expert_results = load_expert_results()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get all experiment folders
    exp_folders = [
        f
        for f in os.listdir(student_policies_dir)
        if os.path.isdir(os.path.join(student_policies_dir, f))
    ]

    # Create color map
    colors = plasma(np.linspace(0, 1, len(exp_folders)))

    # Plot learning curves for each experiment
    for folder, color in zip(sorted(exp_folders), colors):
        print(f"Processing {folder}")
        experiment_dir = os.path.join(student_policies_dir, folder)
        data_dict = get_tb_data(experiment_dir, [attribute])

        x_vals, y_vals = next(iter(data_dict.values()))

        # Apply low-pass filtering if there are enough points
        if len(y_vals) > 10:
            # Replace lowpass_filter with running_average
            window = 5  # Adjust window size as needed
            y_vals = running_average(y_vals, window)
            x_vals = x_vals[window - 1 :]  # Trim x_vals to match y_vals length

        y_vals = to_relative(y_vals, folder, expert_results)

        x_vals = np.insert(x_vals, 0, 0)
        y_vals = np.insert(y_vals, 0, 0)

        # Extract task name from folder name (assuming format like "reach_target_1234")

        # Plot the learning curve with the assigned color
        ax.plot(x_vals, y_vals, label=folder, color=color)

    # Reference line at expert performance
    ax.axhline(100, color="grey", linestyle="--", linewidth=1.5, alpha=0.8, zorder=0)

    # Customize the plot with larger font sizes
    ax.set_xlabel("Steps", fontsize=24)
    ax.set_ylabel("Relative Performance (%)", fontsize=24)
    ax.set_xlim(0, 2e7)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_title("Learning Curves (Single Task)", fontsize=28)
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=20)

    # Convert task names for legend
    handles, labels = ax.get_legend_handles_labels()
    readable_labels = [TASK_NAME_MAPPING.get(label, label) for label in labels]
    ax.legend(
        handles,
        readable_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=18,
    )

    # Save the figure
    out_dir = os.path.join(ROOT_DIR, "data", "figures", "student_policies")
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, "student_policy_single_task_learning_curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Now create the grid of individual plots
    # Create grid of individual plots with 4 rows
    fig, axs = plt.subplots(4, 4, figsize=(24, 24))  # Adjusted figure size for 4x4 grid
    axs = axs.flatten()

    # Hide extra subplots (we only need 14 plots, but have 16 spaces)
    for idx in range(14, 16):
        axs[idx].set_visible(False)

    # Plot each experiment in its own subplot
    for idx, (folder, ax) in enumerate(zip(sorted(exp_folders), axs)):
        print(f"Processing {folder} for individual plot")
        experiment_dir = os.path.join(student_policies_dir, folder)
        data_dict = get_tb_data(experiment_dir, [attribute])

        x_vals, y_vals = next(iter(data_dict.values()))

        # Apply running average if enough points
        if len(y_vals) > 10:
            window = 5
            y_vals = running_average(y_vals, window)
            x_vals = x_vals[window - 1 :]

        y_vals = to_relative(y_vals, folder, expert_results)

        x_vals = np.insert(x_vals, 0, 0)
        y_vals = np.insert(y_vals, 0, 0)

        # Plot in the subplot with thicker line
        ax.plot(x_vals, y_vals, label=folder, linewidth=3.0)
        # Reference line at expert performance
        ax.axhline(100, color="grey", linestyle="--", linewidth=2.0, alpha=0.8, zorder=0)
        ax.set_xlim(0, TASK_X_MAX.get(folder, DEFAULT_X_MAX))
        ax.set_ylim(0, 120)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

        # Use readable task name for title
        readable_name = TASK_NAME_MAPPING.get(folder, folder)
        ax.set_title(readable_name, fontsize=40)

        # Remove grid
        ax.grid(False)

        # Increase tick label size
        ax.tick_params(axis="both", which="major", labelsize=36)

        # Increase scientific notation size
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(36)

        # Add x labels to bottom two rows
        if idx >= 8:  # Bottom two rows get x labels
            ax.set_xlabel("Steps", fontsize=40)

    plt.tight_layout()

    # One shared y label for the whole grid; the per-axes label is too long to
    # repeat on every row without the rows overlapping each other.
    fig.text(
        0.0,
        0.5,
        "Relative Performance (%)",
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=48,
    )

    # Save the grid figure
    out_dir = os.path.join(ROOT_DIR, "data", "figures", "student_policies")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(
        os.path.join(out_dir, "student_policy_single_task_individual_curves.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
