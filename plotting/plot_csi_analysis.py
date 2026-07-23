"""CSI Analysis Plotting Script

This script creates comparison plots for CSI (Control Subspace Identification) 
experiments across multiple tasks, showing the performance of notrain vs finetune approaches.
"""
import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import glob
import json
import os
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from definitions import ROOT_DIR


@lru_cache(maxsize=1)
def load_expert_results():
    """Load the single-task expert benchmark results, keyed by task name."""
    expert_dir = os.path.join(ROOT_DIR, "data", "final_benchmarks", "expert_policies")
    expert_results = {}
    for fname in os.listdir(expert_dir):
        if fname.endswith("_results.json"):
            with open(os.path.join(expert_dir, fname), encoding="utf-8") as file:
                expert_results.update(json.load(file))
    return expert_results


def expert_reference(task, expert_results):
    """Expert avg_solved_step_frac, i.e. the value the CSI curves are normalised by.

    merge_results() summarises every run by avg_solved_step_frac, so the matching
    expert quantity is the same field; the resulting ratio is the one plot_radar.py
    computes from avg_solved_steps.
    """
    expert = expert_results.get(task)
    if expert is None or not expert.get("avg_solved_step_frac"):
        print(f"Warning: no expert avg_solved_step_frac for '{task}'.")
        return None
    return expert["avg_solved_step_frac"]


def _add_vertical_episode_distributions(axis, params):
    """Add 5 individual points from episode sequences.

    Points are plotted slightly offset for notrain (left) and finetune (right).
    """
    task_name = params["task_name"]
    expert_score = params["expert_score"]
    # expert_solved_frac = params["expert_solved_frac"]
    components = params["components"]
    
    notrain_episode_sets = [
        (np.array(result["values"]) / expert_score)
        for result in params["csi_notrain_results"]
    ]

    finetune_episode_sets = [
        (np.array(result["values"]) / expert_score)
        for result in params["csi_finetune_results"]
    ]
    
    bc_finetune_episode_sets = [
        (np.array(result["values"]) / expert_score)
        for result in params["csi_bc_finetune_results"]
    ]
    
    notrain_positions = components[:len(notrain_episode_sets)]
    finetune_positions = components[:len(finetune_episode_sets)] - 0.02 * components.max()
    bc_finetune_positions = components[:len(bc_finetune_episode_sets)] + 0.02 * components.max()
    # Plot 5 points for notrain
    if len(notrain_episode_sets) > 0:
        for i, episode_data in enumerate(notrain_episode_sets):
            if i >= len(notrain_positions):
                continue
            # Select 5 evenly spaced points from the sequence
            selected_points = episode_data[:5]
            x_pos = notrain_positions[i]
            # Add small random jitter to x position for better visibility
            x_jitter = np.random.normal(0, 0.02, len(selected_points))
            axis.scatter(x_pos + x_jitter, selected_points, 
                    color='#111111', alpha=0.6, s=20, edgecolors='#111111', linewidth=0.5)

    # Plot 5 points for finetune
    if len(finetune_episode_sets) > 0:
        for i, episode_data in enumerate(finetune_episode_sets):
            if i >= len(finetune_positions):
                continue
            # Select 5 evenly spaced points from the sequence
            selected_points = episode_data[:5]
            x_pos = finetune_positions[i]
            # Add small random jitter to x position for better visibility
            x_jitter = np.random.normal(0, 0.02, len(selected_points))
            axis.scatter(x_pos + x_jitter, selected_points, 
                    color='#4C72B0', alpha=0.6, s=20, edgecolors='#4C72B0', linewidth=0.5)

    # Plot 5 points for bc-finetune
    if len(bc_finetune_episode_sets) > 0:
        for i, episode_data in enumerate(bc_finetune_episode_sets):
            if i >= len(bc_finetune_positions):
                continue
            selected_points = episode_data[:5]
            x_pos = bc_finetune_positions[i]
            # Add small random jitter to x position for better visibility
            x_jitter = np.random.normal(0, 0.02, len(selected_points))
            axis.scatter(x_pos + x_jitter, selected_points, 
                    color='#C44E52', alpha=0.6, s=20, edgecolors='#C44E52', linewidth=0.5)

# Mapping for better task display names
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

# Metric to use for each task
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
    "kinesis": "solved_steps",
}

def find_exact_task_file(csi_path, prefix, task, csi_dim):
    """Find the exact file for a task, avoiding prefix matching issues."""
    pattern = os.path.join(csi_path, f"{prefix}_{task}{task}*csi_{csi_dim}_results.json")
    files = glob.glob(pattern)
    
    exact_files = []
    for f in files:
        exact_files.append(f)
    
    if len(exact_files) == 0:
        pattern = os.path.join(csi_path, f"{prefix}_{task}*csi_{csi_dim}_results.json")
        files = glob.glob(pattern)
        
        exact_files = []
        for f in files:
            if "baoding_p2_overlap" in f and task == "baoding_p2":
                continue
                
            exact_files.append(f)
    
    # if len(exact_files) == 0:
    #     pattern = os.path.join(csi_path, f"{prefix}_{task}*csi{csi_dim}_results.json")
    #     files = glob.glob(pattern)
        
    #     exact_files = []
    #     for f in files:
    #         exact_files.append(f)
    
    return exact_files[0] if exact_files else None

def load_results(paths):
    """Load JSON results from a list of file paths."""
    results = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as file:
            results.append(json.load(file))
    return results

def get_csi_notrain_paths(task, csi_subspaces):
    """Get paths for CSI expert results for a specific task."""
    base_path = os.path.join(ROOT_DIR, 'data/final_benchmarks_extra/csi_notrain_server')
    csi_dirs = [f'csi{subspace}_all' for subspace in csi_subspaces]
    paths = []
    subspaces = []
    
    for csi_dir, csi_dim in zip(csi_dirs, csi_subspaces):
        csi_path = os.path.join(base_path, csi_dir)
        if os.path.exists(csi_path):
            # Find the exact file for this task
            exact_file = find_exact_task_file(csi_path, "111_gpu_csi_all_bc_student", task, str(csi_dim))
            if exact_file:
                paths.append(exact_file)
                # Extract subspace number from directory name
                subspace_num = int(csi_dir.replace('csi', '').replace('_all', ''))
                subspaces.append(subspace_num)
    return paths, subspaces

def get_csi_finetune_paths(task, csi_subspaces, prefix="555"):
    """Get paths for CSI finetune results for a specific task."""
    base_path = os.path.join(ROOT_DIR, 'data/final_benchmarks_extra/csi_server')
    csi_dirs = [f'csi{subspace}_all' for subspace in csi_subspaces]  # Only these seem to exist for finetune
    paths = []
    subspaces = []
    
    for csi_dir, csi_dim in zip(csi_dirs, csi_subspaces):
        csi_path = os.path.join(base_path, csi_dir)
        if os.path.exists(csi_path):
            # Find the exact file for this task
            exact_file = find_exact_task_file(csi_path, prefix, task, str(csi_dim))
            if exact_file:
                paths.append(exact_file)
                # Extract subspace number from directory name
                subspace_num = int(csi_dir.replace('csi', '').replace('_all', ''))
                subspaces.append(subspace_num)
    return paths, subspaces

def get_csi_bc_finetune_paths(task, csi_subspaces, prefix="666"):
    """Get paths for CSI finetune results for a specific task."""
    base_path = os.path.join(ROOT_DIR, 'data/final_benchmarks_extra/csi_bc_server')
    csi_dirs = [f'csi{subspace}_all' for subspace in csi_subspaces]  # Only these seem to exist for finetune
    paths = []
    subspaces = []
    
    for csi_dir, csi_dim in zip(csi_dirs, csi_subspaces):
        csi_path = os.path.join(base_path, csi_dir)
        if os.path.exists(csi_path):
            # Find the exact file for this task
            exact_file = find_exact_task_file(csi_path, prefix, task, str(csi_dim))
            if exact_file:
                paths.append(exact_file)
                # Extract subspace number from directory name
                subspace_num = int(csi_dir.replace('csi', '').replace('_all', ''))
                subspaces.append(subspace_num)
    return paths, subspaces

def plot_single_task_comparison(
        csi_notrain_results,
        csi_finetune_results,
        csi_bc_finetune_results,
        csi_subspaces,
        task_name,
        axis=None
    ):
    """Create a single task comparison plot."""
    if axis is None:
        _, axis = plt.subplots(figsize=(6, 4))
    # Normalise against the single-task expert. Falling back to the largest CSI
    # subspace without finetuning would instead make every curve end at 1.0 by
    # construction, which says nothing about absolute performance.
    expert_score = expert_reference(task_name, load_expert_results())
    if expert_score is None:
        expert_score = csi_notrain_results[-1]["mean"]

    # Extract data for the specific task
    notrain_solved_fractions = np.array([d["mean"] for d in csi_notrain_results])
    notrain_solved_stds = np.array([d["std"] for d in csi_notrain_results])
    # notrain_solved_errors = np.array([d[task_name][std_key] / np.sqrt(d[task_name]["num_episodes"]) for d in csi_notrain_results])
    finetune_solved_fractions = np.array([d["mean"] for d in csi_finetune_results])
    finetune_solved_stds = np.array([d["std"] for d in csi_finetune_results])
    # finetune_solved_errors = np.array([d[task_name][std_key] / np.sqrt(d[task_name]["num_episodes"]) for d in csi_finetune_results])
    bc_finetune_solved_fractions = np.array([d["mean"] for d in csi_bc_finetune_results])
    bc_finetune_solved_stds = np.array([d["std"] for d in csi_bc_finetune_results])
    
    if bc_finetune_solved_fractions[-1] == 0 and bc_finetune_solved_stds[-1] == 0:
        bc_finetune_solved_fractions[-1] = bc_finetune_solved_fractions[-2]
        bc_finetune_solved_stds[-1] = bc_finetune_solved_stds[-2]
    # bc_finetune_solved_errors = np.array([d[task_name][std_key] / np.sqrt(d[task_name]["num_episodes"]) for d in csi_bc_finetune_results])
    components = np.array(csi_subspaces)
    # Plot CSI Expert performance with error bands (matching PCA style)
    notrain_length = min(
        len(notrain_solved_fractions),
        len(components)
    )
    notrain_norm = notrain_solved_fractions / expert_score
    notrain_std_norm = notrain_solved_stds / expert_score
    axis.plot(components[:notrain_length], notrain_norm[:notrain_length],
              label='CSI w/o Finetune', color='#111111', alpha=0.9,
              marker='o', markersize=3)
    axis.fill_between(components[:notrain_length],
                      notrain_norm[:notrain_length] - notrain_std_norm[:notrain_length],
                      notrain_norm[:notrain_length] + notrain_std_norm[:notrain_length],
                      color='#111111', alpha=0.15)
    
    # Plot CSI Finetune performance with error bands (matching PCA style)
    finetune_length = min(
        len(finetune_solved_fractions),
        len(components)
    )
    finetune_norm = finetune_solved_fractions / expert_score
    finetune_std_norm = finetune_solved_stds / expert_score
    axis.plot(components[:finetune_length], finetune_norm[:finetune_length],
              label='CSI RL Finetune', color='#4C72B0', alpha=0.9,
              marker='o', markersize=3)
    axis.fill_between(components[:finetune_length],
                      finetune_norm[:finetune_length] - finetune_std_norm[:finetune_length],
                      finetune_norm[:finetune_length] + finetune_std_norm[:finetune_length],
                      color='#4C72B0', alpha=0.15)
    
    # Plot CSI BC Finetune performance with error bands (matching PCA style)
    bc_finetune_norm = bc_finetune_solved_fractions / expert_score
    bc_finetune_std_norm = bc_finetune_solved_stds / expert_score
    bc_finetune_length = min(
        len(bc_finetune_solved_fractions),
        len(components)
    )
    axis.plot(components[:bc_finetune_length], bc_finetune_norm[:bc_finetune_length], 
              label='CSI OBC Finetune', color='#C44E52', alpha=0.9,
              marker='o', markersize=3)
    axis.fill_between(components[:bc_finetune_length],
                      bc_finetune_norm[:bc_finetune_length] - bc_finetune_std_norm[:bc_finetune_length],
                      bc_finetune_norm[:bc_finetune_length] + bc_finetune_std_norm[:bc_finetune_length],
                      color='#C44E52', alpha=0.15)
    
    # Add 5 individual points from episode sequences at each component
    _add_vertical_episode_distributions(
        axis=axis,
        params={
            "task_name": task_name,
            "expert_score": expert_score,
            "csi_notrain_results": csi_notrain_results,
            "csi_finetune_results": csi_finetune_results,
            "components": components,
            "csi_bc_finetune_results": csi_bc_finetune_results,
        },
    )

    axis.axhline(y=1.0, color='grey', linestyle='--',
                   alpha=0.5, label='Expert')

    axis.set_xlabel('Number of CSI components')
    axis.set_ylabel('Relative performance (%)')
    axis.set_title(f'{TASK_NAME_MAPPING[task_name]}')
    axis.set_ylim(-0.2, 1.2)  # 1.0 == expert performance
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    
    # Ensure grid is visible
    axis.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    axis.set_axisbelow(True)  # Put grid behind the plot elements
    
    # Add box around each subplot
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)

    return axis

def plot_all_tasks_comparison(all_tasks_data, output_dir='figures', legend_mode='outside'):
    """Create a single figure with subplots for all tasks.

    Args:
        all_tasks_data: Mapping of task name to plotting inputs.
        output_dir: Directory where the figure will be saved.
        legend_mode: Either 'inside' or 'outside' to control legend placement.
    """
    if legend_mode not in {'inside', 'outside'}:
        raise ValueError("legend_mode must be either 'inside' or 'outside'")
    # Set font sizes (matching plot_pca_inactivation.py)
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

    # Calculate subplot layout
    n_tasks = len(all_tasks_data)
    n_cols = 4  # 4 columns
    n_rows = (n_tasks + n_cols - 1) // n_cols  # Ceiling division

    fig_width = 19
    fig_height = (4.0 if legend_mode == 'inside' else 4.2) * n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Flatten axes for easier indexing
    axes_flat = axes.flatten()

    for i, (task_name, task_data) in enumerate(all_tasks_data.items()):
        if i < len(axes_flat):
            csi_notrain_results = task_data['notrain_results']
            csi_finetune_results = task_data['finetune_results']
            csi_bc_finetune_results = task_data['bc_finetune_results']
            csi_subspaces = task_data['subspaces']
            
            plot_single_task_comparison(
                csi_notrain_results,
                csi_finetune_results,
                csi_bc_finetune_results,
                csi_subspaces,
                task_name,
                axes_flat[i]
            )

    # Hide unused subplots
    for i in range(n_tasks, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # The legend sits in the bottom-right corner of the grid, so no column is
    # reserved for it and the panels use the full figure width.
    plt.tight_layout(pad=2.0, w_pad=2.5, h_pad=2.5)

    # Add a single legend positioned to the right of the subplot grid (matching PCA style)
    # Create a dummy plot for the legend
    dummy_lines = []
    dummy_lines.append(plt.Line2D([0], [0], color='#111111', marker='o', linestyle='-', 
                                 markersize=3, label='CSI w/o Finetune'))
    dummy_lines.append(plt.Line2D([0], [0], color='#4C72B0', marker='o', linestyle='-', 
                                 markersize=3, label='CSI RL Finetune'))
    dummy_lines.append(plt.Line2D([0], [0], color='#C44E52', marker='o', linestyle='-', 
                                 markersize=3, label='CSI OBC Finetune'))
    dummy_lines.append(plt.Line2D([0], [0], color='grey', marker='o', linestyle='--', 
                                 markersize=3, label='Expert'))
    
    # Bottom-right of the figure, which falls in the unused subplot slots of the last
    # row, so the legend costs no extra column.
    fig.legend(handles=dummy_lines, loc='lower right', bbox_to_anchor=(0.98, 0.04),
               frameon=True, fancybox=False, shadow=False, facecolor='white')
    output_basename = 'csi_some_tasks_comparison'
    output_path_png = os.path.join(output_dir, f'{output_basename}.png')
    output_path_svg = os.path.join(output_dir, f'{output_basename}.svg')
    # pad_inches keeps the rightmost column's x-label off the crop boundary.
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.savefig(output_path_svg, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

    # print(f"All tasks comparison plot saved to: {output_path}")

def merge_results(default_subspaces, results_list):
    """Merge results from a list of result files."""
    
    merged_tmp = [[] for _ in range(len(default_subspaces))]
    valid_runs = [[0 for _2 in range(len(results_list))] for _ in range(len(default_subspaces))]
    for runid, (results, subspaces) in enumerate(results_list):
        for subspace_result, subspace in zip(results, subspaces):
            for i in range(len(default_subspaces)) :
                if default_subspaces[i] == subspace :
                    merged_tmp[i].append(subspace_result)
                    valid_runs[i][runid] = 1
    merged_result = [{} for _ in range(len(default_subspaces))]
    for i in range(len(default_subspaces)) :
        if len(merged_tmp[i]) > 0 :
            default_key = list(merged_tmp[i][0].keys())[0]
            values = [d[default_key]["avg_solved_step_frac"] for d in merged_tmp[i]]
            merged_result[i] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "num": len(values),
                "examples": [d[default_key]["episode_solve_step_fracs"] for d in merged_tmp[i]],
                "values": values,
                "valid_runs": valid_runs[i]
            }
        else :
            merged_result[i] = {
                "mean": 0,
                "std": 0,
                "num": 0,
                "examples": [],
                "values": [],
                "valid_runs": [0 for _ in range(len(results_list))]
            }
            
    return merged_result

def generate_extra_runs(task, results_list):
    """Generate extra runs for a task."""
    
    all_checkpoints = {
        "elbow_pose": "output/training/ongoing/111_gpu_csi_all_bc_student_elbow_poseelbow_pose_bc_ppo_seed_0_20251009074830/rl_model_5000000_steps.zip",
        "hand_index_reach": "output/training/ongoing/111_gpu_csi_all_bc_student_hand_index_reachhand_index_reach_bc_ppo_seed_0_20251009074508/rl_model_5000000_steps.zip",
        "hand_little_reach": "output/training/ongoing/111_gpu_csi_all_bc_student_hand_little_reachhand_little_reach_bc_ppo_seed_0_20251009074831/rl_model_5000000_steps.zip",
        "hand_middle_reach": "output/training/ongoing/111_gpu_csi_all_bc_student_hand_middle_reachhand_middle_reach_bc_ppo_seed_0_20251009074833/rl_model_5000000_steps.zip",
        "hand_ring_reach": "output/training/ongoing/111_gpu_csi_all_bc_student_hand_ring_reachhand_ring_reach_bc_ppo_seed_0_20251009074833/rl_model_5000000_steps.zip",
        "hand_thumb_reach": "output/training/ongoing/111_gpu_csi_all_bc_student_hand_thumb_reachhand_thumb_reach_bc_ppo_seed_0_20251009074840/rl_model_5000000_steps.zip",
        "kinesis": "output/training/ongoing/111_gpu_csi_all_bc_student_kinesiskinesis_bc_ppo_seed_0_20251009074840/rl_model_5000000_steps.zip",
        "pen": "output/training/ongoing/111_gpu_csi_all_bc_student_penpen_bc_ppo_seed_0_20251009074833/rl_model_5000000_steps.zip",
        "relocate": "output/training/ongoing/111_gpu_csi_all_bc_student_relocaterelocate_bc_ppo_seed_0_20251009074834/rl_model_5000000_steps.zip",
        "reorient": "output/training/ongoing/111_gpu_csi_all_bc_student_reorientreorient_bc_ppo_seed_0_20251009074841/rl_model_5000000_steps.zip",
        "baoding_p1_ccw": "output/training/ongoing/111_gpu_csi_all_bc_student_baoding_p1_ccwbaoding_p1_ccw_bc_ppo_seed_0_20251009074841/rl_model_5000000_steps.zip",
        "baoding_p1_cw": "output/training/ongoing/111_gpu_csi_all_bc_student_baoding_p1_cwbaoding_p1_cw_bc_ppo_seed_0_20251009074842/rl_model_5000000_steps.zip",
        "baoding_p2": "output/training/ongoing/111_gpu_csi_all_bc_student_baoding_p2baoding_p2_bc_ppo_seed_0_20251009074842/rl_model_5000000_steps.zip",
        "baoding_p2_overlap": "output/training/ongoing/111_gpu_csi_all_bc_student_baoding_p2_overlapbaoding_p2_overlap_bc_ppo_seed_0_20251009074844/rl_model_5000000_steps.zip"
    }
    run_prefix = ["666", "777", "888"]
    for result in results_list:
        dim, num, valid_runs = result
        if num < 3 :
            for runid in range(0, 3) :
                if valid_runs[runid] == 0 :
                    command = f"""
                    runai delete job "gpu-csi-bc-finetune-{task.replace('_', '-')}-csi{dim}-seed-{runid+1}"
                    
                    runai submit\\
                    --name "gpu-csi-bc-finetune-{task.replace('_', '-')}-csi{dim}-seed-{runid+1}"\\
                    --image registry.rcp.epfl.ch/arnold/bc\\
                    --run-as-uid 283092\\
                    --run-as-gid 79678\\
                    --gpu 0.3\\
                    --cpu 16 --memory 48Gi --cpu-limit 16 --memory-limit 64Gi\\
                    --existing-pvc claimname=upamathis-scratch,path=/users\\
                    --environment WANDB_API_KEY="c5dcfb879b6cfd9259b7f9e6c0bb6b969dc6f2d3"\\
                    --backoff-limit 0\\
                    --command\\
                    -- python\\
                    "/users/boshi/projects/Arnold/src/main_bc_ppo.py"\\
                    "--project_name=csi_all_bc_finetune"\\
                    "--seed=3"\\
                    "--task={task}"\\
                    "--num_envs=16"\\
                    "--ent_coef=0.0"\\
                    "--vf_coef=0.5"\\
                    "--pg_coef=0.0"\\
                    "--imitation_coef=1.0"\\
                    "--network=csi"\\
                    "--load_path={all_checkpoints[task]}"\\
                    "--load_csi_subspace=output/csi/{task}_csi/subspace.npy"\\
                    "--csi_subspace={dim}"\\
                    "--num_steps=5000000"\\
                    "--out_prefix={run_prefix[runid]}_{task}_csi{dim}_"\\
                    "--device=cuda"\\
                    """
                    print(command)
                    # print(f"Generating extra runs {valid_runs} for {task} with {dim} components...")

def main_all_tasks():
    """Main function to create plots for all tasks in one figure"""
    # Setup
    output_dir = os.path.join(ROOT_DIR, 'data/figures/csi_analysis/')
    os.makedirs(output_dir, exist_ok=True)

    # Set style (matching plot_pca_inactivation.py)
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')
    
    # Set background colors to white
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

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
    
    csi_subspaces = [1, 2, 5, 10, 20, 30, 40]

    # print("Loading results for all tasks...")
    all_tasks_data = {}
    
    for task in tasks:
        # print(f"Processing task: {task}")
        try:
            csi_notrain_paths, csi_notrain_subspaces = get_csi_notrain_paths(task, csi_subspaces)
            csi_finetune_paths_1, csi_finetune_subspaces_1 = get_csi_finetune_paths(task, csi_subspaces, "333")
            csi_finetune_paths_2, csi_finetune_subspaces_2 = get_csi_finetune_paths(task, csi_subspaces, "444")
            csi_finetune_paths_3, csi_finetune_subspaces_3 = get_csi_finetune_paths(task, csi_subspaces, "555")
            csi_bc_finetune_paths_1, csi_bc_finetune_subspaces_1 = get_csi_bc_finetune_paths(task, csi_subspaces, "666")
            csi_bc_finetune_paths_2, csi_bc_finetune_subspaces_2 = get_csi_bc_finetune_paths(task, csi_subspaces, "777")
            csi_bc_finetune_paths_3, csi_bc_finetune_subspaces_3 = get_csi_bc_finetune_paths(task, csi_subspaces, "888")
            
            
            if csi_notrain_paths :
                
                if task == "elbow_pose" :
                    this_csi_subspaces = [1, 2, 5]
                elif "hand" in task :
                    this_csi_subspaces = [1, 2, 5, 10, 20]
                else :
                    this_csi_subspaces = csi_subspaces
                
                # Load results
                csi_notrain_results = load_results(csi_notrain_paths)
                csi_finetune_results_1 = load_results(csi_finetune_paths_1)
                csi_finetune_results_2 = load_results(csi_finetune_paths_2)
                csi_finetune_results_3 = load_results(csi_finetune_paths_3)
                csi_bc_finetune_results_1 = load_results(csi_bc_finetune_paths_1)
                csi_bc_finetune_results_2 = load_results(csi_bc_finetune_paths_2)
                csi_bc_finetune_results_3 = load_results(csi_bc_finetune_paths_3)
                
                csi_finetune_results = merge_results(
                    this_csi_subspaces,
                [
                    (csi_finetune_results_1, csi_finetune_subspaces_1),
                    (csi_finetune_results_2, csi_finetune_subspaces_2),
                    (csi_finetune_results_3, csi_finetune_subspaces_3),
                ])
                csi_bc_finetune_results = merge_results(
                    this_csi_subspaces,
                [
                    (csi_bc_finetune_results_1, csi_bc_finetune_subspaces_1),
                    (csi_bc_finetune_results_2, csi_bc_finetune_subspaces_2),
                    (csi_bc_finetune_results_3, csi_bc_finetune_subspaces_3),
                ])
                csi_notrain_results = merge_results(
                    this_csi_subspaces,
                [
                    (csi_notrain_results, csi_notrain_subspaces),
                ])
                
                all_tasks_data[task] = {
                    'notrain_results': csi_notrain_results,
                    'finetune_results': csi_finetune_results,
                    'bc_finetune_results': csi_bc_finetune_results,
                    'subspaces': this_csi_subspaces
                }
                print("Task:", task)
                print("notrain:", [d["num"] for d in csi_notrain_results])
                print("finetune:", [d["num"] for d in csi_finetune_results])
                print("bc-finetune:", [d["num"] for d in csi_bc_finetune_results])
                
                # generate_extra_runs(task, [(dim, d["num"], d["valid_runs"]) for dim, d in zip(this_csi_subspaces, csi_bc_finetune_results)])
                
                # print(f"  Loaded {len(csi_notrain_results)} notrain, {len(csi_finetune_results)} finetune and {len(csi_bc_finetune_results)} bc-finetune results")
            else:
                pass
                # print(f"  Skipping {task} - missing data files")
        except (FileNotFoundError, KeyError, ValueError) as error:
            print(f"  Error processing {task}: {error}")
            continue

    if all_tasks_data:
        # print(f"Creating combined plot for {len(all_tasks_data)} tasks...")
        plot_all_tasks_comparison(all_tasks_data, output_dir)
        print("Done! Combined plot saved to:", output_dir)
    else:
        pass
        print("No data found for any tasks!")

if __name__ == "__main__":
    # Run the main function to create plots for all tasks in one figure
    main_all_tasks()
