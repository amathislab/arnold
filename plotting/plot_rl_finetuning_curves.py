import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable, Tuple, Dict, Optional, List
from definitions import ROOT_DIR
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.signal import savgol_filter
import itertools

# Define the base directory for fine-tuning experiments
FINETUNE_EXPERIMENTS_BASE_DIR = os.path.join(ROOT_DIR, "data", "expert_policies")

# Base experiment path (student policy)
BASE_EXPERIMENT_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "student_policies",
    "arnold_multi_task",
    "249_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_1",
)

# List of fine-tuning experiment names
FINETUNE_EXPERIMENT_NAMES = [
    "baoding_p2_overlap_262_67874700",
    "baoding_p2_261_64874700",
    "elbow_pose_271_95774700",
    "reorient_258_71274700",
    "kinesis_264_68074700",
]

FINETUNE_EXPERIMENT_LABELS = [
    "Baoding hard",
    "Baoding harder",
    "Elbow pose",
    "Die reorient",
    "Walk to point",
]

# TensorBoard subdirectory name possibilities
TB_SUBDIR_CANDIDATES = ["PPO_0", "tb", "logs", "tensorboard", "rl_finetune"]

# Provided color dictionary (values will be used)
COLOR_LIST = ["#CCB974", "#937860", "#64B5CD", "#55A868", "#4C72B0"]


def get_data_from_tb_log(
    path: str, y_keys: List[str], x_key: str = "step", tb_config: Optional[Dict] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if tb_config is None:
        tb_config = {}

    out_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    try:
        event_acc = EventAccumulator(path, tb_config)
        event_acc.Reload()

        for attr_name in y_keys:
            if attr_name in event_acc.Tags()["scalars"]:
                scalar_events = event_acc.Scalars(attr_name)
                if scalar_events:
                    x_vals, y_vals = np.array(
                        [(getattr(el, x_key), el.value) for el in scalar_events]
                    ).T
                    out_dict[attr_name] = (x_vals, y_vals)
                else:
                    out_dict[attr_name] = (np.array([]), np.array([]))
            else:
                out_dict[attr_name] = (np.array([]), np.array([]))
    except Exception as e:
        print(f"Error reading TensorBoard file {path}: {e}")
        for attr_name in y_keys:
            out_dict[attr_name] = (np.array([]), np.array([]))
    return out_dict


def extend_dict(
    dict1: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dict2: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    for key in dict2.keys():
        if key in dict1 and dict1[key][0].size > 0 and dict2[key][0].size > 0:
            dict1[key] = (
                np.concatenate((dict1[key][0], dict2[key][0])),
                np.concatenate((dict1[key][1], dict2[key][1])),
            )
        elif dict2[key][0].size > 0:
            dict1[key] = dict2[key]
    return dict1


def get_tb_data_for_experiment(
    experiment_log_dir: str, specific_tag: Optional[str] = None
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Optional[str]]:
    all_event_files: List[str] = []
    if os.path.isdir(experiment_log_dir):
        for f_name in os.listdir(experiment_log_dir):
            if f_name.startswith("events.out.tfevents"):
                all_event_files.append(os.path.join(experiment_log_dir, f_name))

    for tb_subdir_name in TB_SUBDIR_CANDIDATES:
        tb_dir_path = os.path.join(experiment_log_dir, tb_subdir_name)
        if os.path.isdir(tb_dir_path):
            for f_name in os.listdir(tb_dir_path):
                if f_name.startswith("events.out.tfevents"):
                    all_event_files.append(os.path.join(tb_dir_path, f_name))

    all_event_files = sorted(list(set(all_event_files)))

    if not all_event_files:
        # print(f"Warning: No TensorBoard event files found for {experiment_log_dir}.")
        return {}, specific_tag if specific_tag else None

    identified_tag_to_use = specific_tag
    if not identified_tag_to_use:
        try:
            # Load only scalars, minimal size_guidance for just listing tags
            event_acc = EventAccumulator(
                all_event_files[0], size_guidance={"scalars": 0}
            )
            event_acc.Reload()
            for tag_candidate in event_acc.Tags()["scalars"]:
                if tag_candidate.endswith("/solved"):
                    identified_tag_to_use = tag_candidate
                    # print(f"Dynamically found tag '{identified_tag_to_use}' for {os.path.basename(experiment_log_dir)}")
                    break
            if not identified_tag_to_use:  # Check if a tag was found
                print(
                    f"Warning: No tag ending with '/solved' found in {all_event_files[0]} for {os.path.basename(experiment_log_dir)}"
                )
                return {}, None  # Return None for tag if not found
        except Exception as e:
            print(
                f"Error reading tags from {all_event_files[0]} for {os.path.basename(experiment_log_dir)}: {e}"
            )
            return {}, None  # Return None for tag on error

    # print(f"Using tag '{identified_tag_to_use}' for {os.path.basename(experiment_log_dir)}")

    data_dict: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # Ensure identified_tag_to_use is not None before proceeding
    if not identified_tag_to_use:
        # This case should be rare if the above logic correctly returns None for the tag
        # print(f"Error: identified_tag_to_use is None before data extraction for {os.path.basename(experiment_log_dir)}")
        return {}, None

    for tb_file_path in all_event_files:
        # Pass a list with the single identified tag
        data_dict_to_add = get_data_from_tb_log(tb_file_path, [identified_tag_to_use])
        data_dict = extend_dict(data_dict, data_dict_to_add)

    # Sort and clean up
    if identified_tag_to_use in data_dict:
        if data_dict[identified_tag_to_use][0].size > 0:
            sort_idx = np.argsort(data_dict[identified_tag_to_use][0])
            data_dict[identified_tag_to_use] = (
                data_dict[identified_tag_to_use][0][sort_idx],
                data_dict[identified_tag_to_use][1][sort_idx],
            )
        else:
            # Tag was found, but no data points for it across all files
            del data_dict[identified_tag_to_use]
            # print(f"Info: No data points for attribute {identified_tag_to_use} in {experiment_log_dir}, though tag was found.")

    if not data_dict or identified_tag_to_use not in data_dict:
        # This means either data_dict is empty or the identified_tag is not in it
        return (
            {},
            identified_tag_to_use,
        )  # Return empty dict but the tag name if it was initially found

    return data_dict, identified_tag_to_use


def plot_learning_curves():
    fig, ax = plt.subplots(figsize=(6, 4))  # Increased figure size
    color_cycler = itertools.cycle(COLOR_LIST)
    base_label_added = False  # Flag to ensure base label is added only once

    for i, ft_exp_name in enumerate(FINETUNE_EXPERIMENT_NAMES):
        current_color = next(color_cycler)
        current_label = FINETUNE_EXPERIMENT_LABELS[i]  # Use label from the list
        ft_experiment_dir = os.path.join(FINETUNE_EXPERIMENTS_BASE_DIR, ft_exp_name)

        if not os.path.isdir(ft_experiment_dir):
            print(
                f"Warning: Fine-tuning experiment directory not found: {ft_experiment_dir}"
            )
            continue

        print(f"Processing fine-tuning experiment: {ft_exp_name}")

        # 1. Get data for fine-tuning experiment to identify the tag
        # We only need the tag name from the fine-tuning experiment's perspective first
        _, solved_tag_name = get_tb_data_for_experiment(ft_experiment_dir)

        if not solved_tag_name:
            print(f"Could not identify a '/solved' tag for {ft_exp_name}. Skipping.")
            continue

        print(
            f"Identified tag '{solved_tag_name}' for {ft_exp_name}. Now processing base and fine-tune data."
        )

        # 2. Get data for the BASE experiment using the identified solved_tag_name
        base_data_dict, _ = get_tb_data_for_experiment(
            BASE_EXPERIMENT_PATH, specific_tag=solved_tag_name
        )

        last_base_step = 0
        plotted_base = False
        if (
            base_data_dict
            and solved_tag_name in base_data_dict
            and base_data_dict[solved_tag_name][0].size > 0
        ):
            base_x_vals, base_y_vals = base_data_dict[solved_tag_name]
            if len(base_y_vals) > 101:  # Window length for savgol_filter
                base_y_vals_smooth = savgol_filter(
                    base_y_vals, window_length=101, polyorder=2
                )
            else:
                base_y_vals_smooth = base_y_vals  # Not enough points to smooth

            label_for_base = None
            if not base_label_added:
                label_for_base = "OBC Multi-task"
                base_label_added = True

            ax.plot(
                base_x_vals,
                base_y_vals_smooth,
                label=label_for_base,  # Use conditional label
                color="black",
                alpha=0.2,
                linewidth=2,
            )
            last_base_step = base_x_vals[-1] if base_x_vals.size > 0 else 0
            plotted_base = True
            print(
                f"  Plotted base data for {solved_tag_name} from {BASE_EXPERIMENT_PATH} (up to step {last_base_step})"
            )
        else:
            print(
                f"  No data for tag '{solved_tag_name}' found in base experiment: {BASE_EXPERIMENT_PATH}"
            )

        # 3. Get data for the FINE-TUNING experiment again, this time we know the tag
        ft_data_dict, _ = get_tb_data_for_experiment(
            ft_experiment_dir, specific_tag=solved_tag_name
        )

        if (
            ft_data_dict
            and solved_tag_name in ft_data_dict
            and ft_data_dict[solved_tag_name][0].size > 0
        ):
            ft_x_vals, ft_y_vals = ft_data_dict[solved_tag_name]

            # Shift fine-tuning steps
            # Assuming fine-tuning logs restart step counts. If they continue, this needs adjustment.
            ft_x_vals_shifted = ft_x_vals  # Removed + last_base_step

            if len(ft_y_vals) > 101:  # Window length for savgol_filter
                ft_y_vals_smooth = savgol_filter(
                    ft_y_vals, window_length=101, polyorder=2
                )
            else:
                ft_y_vals_smooth = ft_y_vals  # Not enough points to smooth

            # Use the fine-tuning experiment name for the label, only add label if we plot this part

            ax.plot(
                ft_x_vals_shifted,
                ft_y_vals_smooth,
                label="PPO " + current_label,  # Use predefined label
                color=current_color,
                alpha=1.0,
                linewidth=2.5,
            )
            print(
                f"  Plotted fine-tuning data for {solved_tag_name} from {ft_exp_name} (shifted by {last_base_step} steps)"
            )
        elif plotted_base:  # If base was plotted but no FT data for this tag
            # Add label to the base segment for clarity, using an invisible plot for the legend entry
            ax.plot(
                [], [], label=current_label + " (Base Only)", color=current_color
            )  # Use predefined label
            print(
                f"  No fine-tuning data for tag '{solved_tag_name}' in {ft_exp_name}, but base was plotted."
            )
        else:  # Neither base nor FT data for this tag
            print(
                f"  No data for tag '{solved_tag_name}' could be plotted for fine-tuning experiment {ft_exp_name}."
            )

    ax.set_xlabel("Training steps")
    ax.set_ylabel("Solved fraction")
    ax.set_title("OBC + RL fine-tuning")
    ax.legend(loc="best", fontsize="small", ncol=2)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_xlim(right=68e6)  # Set x-axis limit to 68M
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    out_dir = os.path.join(ROOT_DIR, "data", "figures", "rl_finetuning_combined")
    os.makedirs(out_dir, exist_ok=True)
    output_filename = "rl_finetuning_combined_solved_curves.png"
    fig.savefig(os.path.join(out_dir, output_filename), dpi=300, bbox_inches="tight")
    fig.savefig(
        os.path.join(out_dir, output_filename.replace(".png", ".svg")),
        bbox_inches="tight",
    )
    print(f"Plot saved to {os.path.join(out_dir, output_filename)}")


if __name__ == "__main__":
    plot_learning_curves()
