import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from definitions import ROOT_DIR
from scipy.signal import savgol_filter
from plot_csi_analysis import TASK_NAME_MAPPING

# Tasks whose "solved" scalar is not logged (it is all-NaN), so the episode reward is
# read instead and normalised against the expert's average cumulative reward.
REWARD_METRIC_TASKS = {"kinesis"}


def load_expert_results():
    """Load the single-task expert benchmark results, keyed by task name."""
    expert_dir = os.path.join(ROOT_DIR, "data", "final_benchmarks", "expert_policies")
    expert_results = {}
    for fname in os.listdir(expert_dir):
        if fname.endswith("_results.json"):
            with open(os.path.join(expert_dir, fname)) as f:
                expert_results.update(json.load(f))
    return expert_results


def expert_reference(task, expert_results):
    """Return the expert value a task's logged curve should be divided by.

    The logged "solved" scalar is the fraction of solved steps per episode, so the
    matching expert quantity is avg_solved_step_frac; dividing by it yields the same
    student/expert ratio that plot_radar.py computes from avg_solved_steps.
    """
    expert = expert_results.get(task)
    if expert is None:
        print(f"Warning: no expert results for '{task}', curve left unnormalised.")
        return None
    key = "avg_cum_reward" if task in REWARD_METRIC_TASKS else "avg_solved_step_frac"
    reference = expert.get(key)
    if not reference:
        print(f"Warning: expert '{key}' is missing or zero for '{task}'.")
        return None
    return reference

def find_finetune_curve_file(
        base_path,
        prefix,
        task,
        csi_dim
    ) :
    # output/training/server/ongoing/222_baoding_p1_ccw_csi1_baoding_p1_ccw_mlp_ppo_seed_0_20251014092354
    pattern = os.path.join(base_path, f"{prefix}_{task}_csi{csi_dim}_*", "PPO_0", "events.out.tfevents.*")
    files = glob.glob(pattern)
    # assert len(files) == 1, f"Expected 1 file, got {len(files)}"
    if len(files) == 0:
        return None
    return files[0]

def find_training_curve_file(
        base_path,
        prefix,
        task,
    ) :
    pattern = os.path.join(base_path, f"{prefix}_gpu_csi_all_bc_student_{task}*", "PPO_0", "events.out.tfevents.*")
    files = glob.glob(pattern)
    # assert len(files) == 1, f"Expected 1 file, got {len(files)}"
    if len(files) == 0:
        return None
    return files[0]

def plot_all_tasks_comparison(all_tasks_data) :
    '''
    Example:
    {
        "elbow_pose": {
            "csi_subspaces": [1, 2, 5],
            "training": "output/training/server/ongoing/111_gpu_csi_all_bc_student_elbow_pose*",
            "rl_finetune": [
                "output/training/server/ongoing/222_gpu_csi_all_bc_student_elbow_pose_csi1*",
                "output/training/server/ongoing/222_gpu_csi_all_bc_student_elbow_pose_csi2*",
                "output/training/server/ongoing/222_gpu_csi_all_bc_student_elbow_pose_csi5*",
            ],
            "bc_finetune": [
                "output/training/server/ongoing/666_gpu_csi_all_bc_student_hand_index_reach_csi1*",
                "output/training/server/ongoing/666_gpu_csi_all_bc_student_hand_index_reach_csi2*",
                "output/training/server/ongoing/666_gpu_csi_all_bc_student_hand_index_reach_csi5*",
            ]
        }
    }
    '''
    # Helper to load a single scalar series (prefer tags ending with '/solved')
    def _load_scalar_from_event_file(event_file_path) :
        try:
            event_acc = EventAccumulator(event_file_path, size_guidance={"scalars": 0})
            event_acc.Reload()
            tags = event_acc.Tags().get("scalars", [])
            chosen_tag = None
            # Prefer tags ending with '/solved'
            for tag in tags:
                if str(tag).endswith("/solved"):
                    chosen_tag = tag
                    break
            # Fallback: any tag containing 'solved'
            if chosen_tag is None:
                for tag in tags:
                    if "solved" in str(tag).lower():
                        chosen_tag = tag
                        break
            if chosen_tag is None:
                return None, None
            scalar_events = event_acc.Scalars(chosen_tag)
            if not scalar_events:
                return None, None
            if "kinesis" in event_file_path:
                scalar_events = event_acc.Scalars("rollout/ep_rew_mean")
            x_values = [ev.step for ev in scalar_events]
            y_values = [ev.value for ev in scalar_events]
            return x_values, y_values
        except (OSError, ValueError, RuntimeError) as err:
            print(f"Failed to read {event_file_path}: {err}")
            return None, None

    # Font sizes
    small_size = 16
    medium_size = 20
    bigger_size = 24

    plt.rc('font', size=small_size)
    plt.rc('axes', titlesize=bigger_size)
    plt.rc('axes', labelsize=medium_size)
    plt.rc('xtick', labelsize=small_size)
    plt.rc('ytick', labelsize=small_size)
    plt.rc('legend', fontsize=medium_size)
    plt.rc('figure', titlesize=bigger_size)

    # Create fixed 4x4 grid
    n_rows, n_cols = 4, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 16))
    axes_flat = axes.flatten()

    # Colors
    color_training = '#000000'  # black
    color_rl = '#4C72B0'        # red-ish
    color_bc = '#C44E52'        # yellow-ish

    def _smooth_series(y_values) :
        try:
            n = len(y_values)
            if n < 7:
                return y_values
            window = min(101, n if n % 2 == 1 else n - 1)
            if window < 7:
                window = 7
            if window % 2 == 0:
                window -= 1
            return savgol_filter(y_values, window_length=window, polyorder=2)
        except Exception:
            return y_values

    expert_results = load_expert_results()

    # Plot each task
    plotted_count = 0
    for task_name, data in all_tasks_data.items() :
        if plotted_count >= len(axes_flat):
            break
        ax = axes_flat[plotted_count]
        ax.set_title(TASK_NAME_MAPPING[task_name])
        ax.grid(True, alpha=0.3)

        # Every curve in this panel is expressed as a percentage of the single-task
        # expert, so that panels are comparable across tasks.
        expert_ref = expert_reference(task_name, expert_results)

        def _relative(y_smooth) :
            if expert_ref is None:
                return np.asarray(y_smooth, dtype=float)
            return np.asarray(y_smooth, dtype=float) / expert_ref * 100

        # Training (single file)
        train_path = data.get("training")
        label_added_training = False
        if isinstance(train_path, str) and os.path.exists(train_path):
            x_values, y_values = _load_scalar_from_event_file(train_path)
            if x_values is not None and y_values is not None and len(x_values) > 0:
                y_smooth = _smooth_series(y_values)
                ax.plot(x_values, _relative(y_smooth), color=color_training, alpha=0.25, linewidth=2.0, label='Training' if not label_added_training else None)
                label_added_training = True


        # RL finetune (list of files)
        rl_list = data.get("rl_finetune", []) or []
        label_added_rl = False
        rl_count = len(rl_list)
        for rl_idx, path in enumerate(rl_list):
            if isinstance(path, str) and os.path.exists(path):
                x_values, y_values = _load_scalar_from_event_file(path)
                if x_values is None or y_values is None or len(x_values) == 0:
                    continue
                y_smooth = _smooth_series(y_values)
                if rl_count <= 1:
                    alpha_val = 1.0
                else:
                    alpha_val = 0.3 + 0.7 * (rl_idx / (rl_count - 1))
                ax.plot(x_values, _relative(y_smooth), color=color_rl, alpha=alpha_val, linewidth=1.5, label='RL finetune' if not label_added_rl else None)
                label_added_rl = True

        # BC finetune (list of files)
        bc_list = data.get("bc_finetune", []) or []
        label_added_bc = False
        bc_count = len(bc_list)
        for bc_idx, path in enumerate(bc_list):
            if isinstance(path, str) and os.path.exists(path):
                x_values, y_values = _load_scalar_from_event_file(path)
                if x_values is None or y_values is None or len(x_values) == 0:
                    continue
                y_smooth = _smooth_series(y_values)
                if bc_count <= 1:
                    alpha_val = 1.0
                else:
                    alpha_val = 0.3 + 0.7 * (bc_idx / (bc_count - 1))
                ax.plot(x_values, _relative(y_smooth), color=color_bc, alpha=alpha_val, linewidth=1.5, label='OBC finetune' if not label_added_bc else None)
                label_added_bc = True

        ax.axvline(5_000_000, color='grey', linestyle=':', linewidth=1.2, alpha=0.7)
        # Expert performance
        ax.axhline(100, color='grey', linestyle='--', linewidth=1.2, alpha=0.8, zorder=0)

        ax.set_xlabel('Training steps')
        ax.set_ylabel('Relative performance (%)')
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        # Put grid behind
        ax.set_axisbelow(True)

        # Add box
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('black')
            spine.set_linewidth(0.8)
        plotted_count += 1

    # Hide remaining axes if any
    for j in range(plotted_count, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Single shared legend
    handles, labels = [], []
    for ax in reversed(axes_flat):
        h, l = ax.get_legend_handles_labels()
        for hh, ll in reversed(list(zip(h, l))):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
        fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.98, 0.02), frameon=True)

    plt.tight_layout()
    out_dir = os.path.join(ROOT_DIR, 'data', 'figures', 'csi_analysis')
    os.makedirs(out_dir, exist_ok=True)
    out_basename = 'csi_learning_curves_grid'
    out_path_png = os.path.join(out_dir, f'{out_basename}.png')
    out_path_svg = os.path.join(out_dir, f'{out_basename}.svg')
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_path_svg, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined CSI learning curves to: {out_path_png} and {out_path_svg}")

if __name__ == "__main__" :
    
    tasks = [
        "elbow_pose",
        "hand_index_reach",
        "hand_little_reach",
        "hand_middle_reach",
        "hand_ring_reach",
        "hand_thumb_reach",
        "kinesis",
        "pen",
        "relocate",
        "reorient",
        "baoding_p1_ccw",
        "baoding_p1_cw",
        "baoding_p2",
        "baoding_p2_overlap",
    ]
    csi_dims = [1, 2, 5, 10, 20, 30, 40]
    root = "output/training/server/ongoing"

    # file = find_finetune_curve_file(
    #     base_path = root,
    #     prefix = "222",
    #     task = tasks[0],
    #     csi_dim = csi_dims[0],
    # )
    # print(file)
    
    # Set style (matching plot_pca_inactivation.py)
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')

    # Set background colors to white
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

    # Get data
    all_tasks_data = {}
    
    for task in tasks :
        
        all_tasks_data[task] = {}
        
        if task == "elbow_pose" :
            this_csi_subspaces = [1, 2, 5]
        elif "hand" in task :
            this_csi_subspaces = [1, 2, 5, 10, 20]
        else :
            this_csi_subspaces = csi_dims
            
        all_tasks_data[task]["csi_subspaces"] = this_csi_subspaces
        
        training_curve_file = find_training_curve_file(
            base_path = root,
            prefix = "111",
            task = task,
        )
        all_tasks_data[task]["training"] = training_curve_file
        all_tasks_data[task]["rl_finetune"] = []
        all_tasks_data[task]["bc_finetune"] = []
        
        for csi_dim in this_csi_subspaces :
            rl_finetune_curve_file = find_finetune_curve_file(
                base_path = root,
                prefix = "555",
                task = task,
                csi_dim = csi_dim,
            )
            if rl_finetune_curve_file is not None :
                all_tasks_data[task]["rl_finetune"].append(rl_finetune_curve_file)
        for csi_dim in this_csi_subspaces :
            bc_finetune_curve_file = find_finetune_curve_file(
                base_path = root,
                prefix = "666",
                task = task,
                csi_dim = csi_dim,
            )
            if bc_finetune_curve_file is not None :
                all_tasks_data[task]["bc_finetune"].append(bc_finetune_curve_file)

    plot_all_tasks_comparison(all_tasks_data)
