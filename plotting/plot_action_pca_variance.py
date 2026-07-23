import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import argparse
import glob
import os
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns  # Added for styling consistency

# Define font sizes from plot_pca_inactivation.py
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

# Apply general style from plot_pca_inactivation.py
plt.style.use("default")
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.color"] = "#cccccc"
# sns.set_palette("husl") # Not strictly needed as we define colors directly

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


def load_actions_from_h5(h5_path):
    """Load actions from an H5 file"""
    with h5py.File(h5_path, "r") as f:
        actions = np.array(f["action_means"])
        # solved = np.array(f["solved"]) # Not needed for this script

    # Actions are typically (num_steps, 1, action_dim), squeeze to (num_steps, action_dim)
    if actions.ndim == 3 and actions.shape[1] == 1:
        actions = actions.squeeze(axis=1)
    elif actions.ndim != 2:
        raise ValueError(
            f"Unexpected action dimensions: {actions.shape} in file {h5_path}"
        )
    return actions


def load_all_episode_actions(base_dir, task):
    """Load actions from all episodes for a given task"""
    pattern = os.path.join(base_dir, f"{task}_episode_*.h5")
    files = glob.glob(pattern)

    if not files:
        print(f"No episode files found for task {task} with pattern {pattern}")
        return np.array([])

    all_actions = []
    for f_path in files:
        try:
            actions = load_actions_from_h5(f_path)
            if (
                actions.ndim == 2 and actions.shape[0] > 0
            ):  # Ensure actions are 2D and not empty
                all_actions.append(actions)
            elif actions.size == 0:
                print(f"Warning: No actions loaded from {f_path}")
            else:
                print(
                    f"Warning: Actions from {f_path} have unexpected shape or type: {actions.shape}"
                )

        except Exception as e:
            print(f"Error loading actions from {f_path}: {e}")
            continue

    if not all_actions:
        print(f"No valid actions loaded for task {task}.")
        return np.array([])

    return np.vstack(all_actions)


def _plot_pca_data_set(
    ax, task_data_dict, max_components, avg_label, avg_color, ind_color
):
    """Helper function to plot one set of PCA data (average and individual lines)."""
    if not task_data_dict:
        print(f"No data for {avg_label}")
        return 0, np.zeros(max_components)  # Return 0 tasks processed and zero array

    sum_cum_var = np.zeros(max_components)
    num_tasks_for_avg = 0
    component_nums_for_avg = np.arange(1, max_components + 1)

    for task_name, data in task_data_dict.items():
        cum_var = data["cum_var"]
        if cum_var.size == 0:
            continue

        current_components = np.arange(1, len(cum_var) + 1)
        ax.plot(current_components, cum_var, color=ind_color, alpha=0.15, linewidth=1.5)

        padded_cum_var = np.pad(
            cum_var,
            (0, max_components - len(cum_var)),
            "constant",
            constant_values=(cum_var[-1] if cum_var.size > 0 else 0),
        )
        sum_cum_var += padded_cum_var
        num_tasks_for_avg += 1

    if num_tasks_for_avg > 0:
        avg_cum_var = sum_cum_var / num_tasks_for_avg
        ax.plot(
            component_nums_for_avg,
            avg_cum_var,
            color=avg_color,
            linewidth=3,
            label=avg_label,
            alpha=0.9,
        )
        return num_tasks_for_avg, avg_cum_var
    else:
        print(f"No tasks to average for {avg_label}")
        return 0, np.zeros(max_components)


def plot_comparison_cumulative_variance(
    task_data_separate,
    task_data_shared,
    title,
    out_path,
    label_separate,
    color_separate,
    label_shared,
    color_shared,
):
    """Plot a comparison of two sets of PCA cumulative variance data on the same plot."""
    # Apply font sizes
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, ax = plt.subplots(figsize=(8, 6))  # Match plot_merged_summary

    all_cum_vars_separate = [
        data["cum_var"]
        for data in task_data_separate.values()
        if data["cum_var"].size > 0
    ]
    all_cum_vars_shared = [
        data["cum_var"]
        for data in task_data_shared.values()
        if data["cum_var"].size > 0
    ]

    if not all_cum_vars_separate and not all_cum_vars_shared:
        print(f"No valid cumulative variance data for plotting: {title}")
        plt.close(fig)
        return

    max_components = 0
    for cum_var_array_list in [all_cum_vars_separate, all_cum_vars_shared]:
        for cum_var_array in cum_var_array_list:
            if len(cum_var_array) > max_components:
                max_components = len(cum_var_array)

    if max_components == 0:
        print(f"Max components is 0, cannot plot: {title}")
        plt.close(fig)
        return

    # Plot Task-Specific PCA Data
    _plot_pca_data_set(
        ax,
        task_data_separate,
        max_components,
        label_separate,
        color_separate,
        color_separate,  # ind_color is same as avg_color for consistency
    )

    # Plot Shared PCA Data
    _plot_pca_data_set(
        ax,
        task_data_shared,
        max_components,
        label_shared,
        color_shared,
        color_shared,  # ind_color is same as avg_color for consistency
    )

    # Removed threshold lines
    # plt.axhline(y=0.85, color="r", linestyle="--", alpha=0.5, label="85% threshold")
    # plt.axhline(y=0.95, color="g", linestyle="--", alpha=0.5, label="95% threshold")

    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative EV")
    ax.set_title(title)
    ax.legend(loc="lower right")  # Match plot_merged_summary
    ax.grid(True, alpha=0.3)  # Already set by rcParams but good to be explicit
    ax.set_ylim((-0.05, 1.05))  # Ensure plot range is good
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.savefig(out_path.replace(".png", ".svg"), bbox_inches="tight")
    print(f"Saved plot to {out_path}")
    plt.close(fig)


def plot_individual_task_variance_comparison(
    task_data_separate, task_data_shared, out_dir, task_name_mapping, tasks_to_plot
):
    """
    Generate individual plots for each task, comparing cumulative variance from
    task-specific PCA vs. projection onto shared PCA.
    """
    # Apply font sizes
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

    print("\nGenerating individual task variance comparison plots...")

    for task_name in tasks_to_plot:
        if task_name not in task_data_separate and task_name not in task_data_shared:
            print(f"No data for task {task_name}, skipping individual plot.")
            continue

        fig, ax = plt.subplots(figsize=(6, 4.2))  # Match plot_individual_tasks

        # Plot task-specific PCA
        if task_name in task_data_separate:
            data_s = task_data_separate[task_name]
            cum_var_s = data_s["cum_var"]
            if cum_var_s.size > 0:
                components_s = np.arange(1, len(cum_var_s) + 1)
                ax.plot(
                    components_s,
                    cum_var_s,
                    label="Task-specific PCA",
                    color="#4C72B0",  # Blueish
                    alpha=0.9,
                    linewidth=2.5,
                )
        else:
            print(f"No separate PCA data for task {task_name}")

        # Plot shared PCA projection for this task
        if task_name in task_data_shared:
            data_sh = task_data_shared[task_name]
            cum_var_sh = data_sh["cum_var"]
            if cum_var_sh.size > 0:
                components_sh = np.arange(1, len(cum_var_sh) + 1)
                ax.plot(
                    components_sh,
                    cum_var_sh,
                    label="Shared PCA projection",
                    color="#C44E52",  # Reddish
                    alpha=0.9,
                    linewidth=2.5,
                )
        else:
            print(f"No shared PCA data for task {task_name}")

        plot_title = task_name_mapping.get(
            task_name, task_name.replace("_", " ").title()
        )
        ax.set_title(plot_title)
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Cumulative EV")
        # ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim((-0.05, 1.05))

        plt.tight_layout()
        out_path_task = os.path.join(
            out_dir, f"action_variance_comparison_{task_name}.png"
        )
        plt.savefig(out_path_task, bbox_inches="tight", dpi=300)
        plt.savefig(out_path_task.replace(".png", ".svg"), bbox_inches="tight")
        print(f"Saved individual plot for {task_name} to {out_path_task}")
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze and plot PCA cumulative explained variance for actions."
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        required=True,
        help="Directory containing H5 activation files.",
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Directory to save output plots."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "hand_thumb_reach",
            "hand_index_reach",
            "hand_middle_reach",
            "reorient",
            "hand_ring_reach",
            "hand_little_reach",
            "pen",
            "baoding_p1_ccw",
            "baoding_p1_cw",
            "baoding_p2",
            "baoding_p2_overlap",
        ],
        help="List of tasks to analyze.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_task_actions_data = {}
    print("Loading actions for all tasks...")
    for task in args.tasks:
        print(f"Loading actions for task: {task}")
        actions = load_all_episode_actions(args.activations_dir, task)
        if actions.size > 0 and actions.ndim == 2:
            all_task_actions_data[task] = actions
        else:
            print(f"No valid actions loaded for task {task}, skipping.")

    if not all_task_actions_data:
        print("No actions loaded for any task. Exiting.")
        return

    # 1. Separate PCA analysis
    task_pcas_for_plot_separate = {}
    print("\nAnalyzing with separate PCAs...")
    for task_name, actions in all_task_actions_data.items():
        if actions.shape[0] < actions.shape[1]:
            print(
                f"Skipping task {task_name} for separate PCA: not enough samples ({actions.shape[0]}) for features ({actions.shape[1]})"
            )
            continue
        try:
            pca = PCA()
            pca.fit(actions)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            task_pcas_for_plot_separate[task_name] = {"pca": pca, "cum_var": cum_var}
        except Exception as e:
            print(f"Error computing separate PCA for task {task_name}: {e}")

    # 2. Shared PCA analysis
    task_pcas_for_plot_shared = {}
    print("\nAnalyzing with shared PCA...")

    # Filter out tasks with no actions or actions with incompatible shapes for vstack
    valid_actions_list = [
        acts
        for acts in all_task_actions_data.values()
        if acts.ndim == 2 and acts.shape[0] > 0
    ]

    if not valid_actions_list:
        print(
            "No valid actions available from any task for shared PCA. Skipping shared PCA analysis."
        )
        return

    # Determine common number of features (action_dim)
    action_dims = [acts.shape[1] for acts in valid_actions_list]
    if not all(d == action_dims[0] for d in action_dims):
        print("Action dimensions mismatch between tasks. Cannot perform shared PCA.")
        # Optionally, print which tasks have different dimensions
        for task_name, actions_arr in all_task_actions_data.items():
            if actions_arr.ndim == 2 and actions_arr.shape[0] > 0:
                print(f"Task {task_name} action dim: {actions_arr.shape[1]}")
        return

    all_actions_combined = np.vstack(valid_actions_list)

    if all_actions_combined.shape[0] < all_actions_combined.shape[1]:
        print(
            f"Skipping shared PCA: not enough combined samples ({all_actions_combined.shape[0]}) for features ({all_actions_combined.shape[1]})"
        )
    elif all_actions_combined.size == 0:
        print("Combined actions array is empty. Skipping shared PCA.")
    else:
        try:
            shared_pca_model = PCA()
            shared_pca_model.fit(all_actions_combined)

            for task_name, actions in all_task_actions_data.items():
                if (
                    actions.size == 0 or actions.ndim != 2
                ):  # Already checked, but good for safety
                    print(
                        f"Skipping task {task_name} for shared PCA projection: no valid actions."
                    )
                    continue
                if actions.shape[1] != shared_pca_model.n_features_in_:
                    print(
                        f"Skipping task {task_name} for shared PCA projection: action dimension mismatch ({actions.shape[1]} vs {shared_pca_model.n_features_in_})."
                    )
                    continue

                transformed_task_data = shared_pca_model.transform(actions)

                task_total_variance = np.sum(np.var(transformed_task_data, axis=0))
                if task_total_variance < 1e-9:  # Check for near-zero variance
                    print(
                        f"Task {task_name} has zero or near-zero variance after projection with shared PCA. Assigning zero variance ratios."
                    )
                    var_ratios = np.zeros(shared_pca_model.n_components_)
                else:
                    var_ratios = (
                        np.var(transformed_task_data, axis=0) / task_total_variance
                    )

                cum_var_shared_for_task = np.cumsum(var_ratios)
                task_pcas_for_plot_shared[task_name] = {
                    "pca": shared_pca_model,
                    "cum_var": cum_var_shared_for_task,
                }
        except Exception as e:
            print(f"Error during shared PCA computation or projection: {e}")

    # 3. Plot combined comparison
    if task_pcas_for_plot_separate or task_pcas_for_plot_shared:
        plot_comparison_cumulative_variance(
            task_pcas_for_plot_separate,
            task_pcas_for_plot_shared,
            "Action Cumulative Variance Comparison",
            os.path.join(args.out_dir, "action_variance_comparison_merged.png"),
            label_separate="Task-specific PCA (mean)",
            color_separate="#4C72B0",  # Blueish
            label_shared="Shared PCA (mean)",
            color_shared="#C44E52",  # Reddish
        )
        plot_individual_task_variance_comparison(
            task_pcas_for_plot_separate,
            task_pcas_for_plot_shared,
            args.out_dir,
            TASK_NAME_MAPPING,
            args.tasks,  # Pass the list of tasks processed
        )
    else:
        print("No data available for plotting comparison.")

    # 4. Plot individual task comparisons
    if task_pcas_for_plot_separate or task_pcas_for_plot_shared:
        print("\nPlotting individual task comparisons...")
        plot_individual_task_variance_comparison(
            task_pcas_for_plot_separate,
            task_pcas_for_plot_shared,
            args.out_dir,
            TASK_NAME_MAPPING,
            tasks_to_plot=args.tasks,
        )
    else:
        print("No data available for plotting individual task comparisons.")


if __name__ == "__main__":
    main()


"""
python src/plot_action_pca_variance.py --activations_dir data/activations/285_64670238 --out_dir data/figures/cumulative_variance/285_64670238
"""
