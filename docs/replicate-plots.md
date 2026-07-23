# Replicate plots

Scripts for the paper's performance plots and learning curves. All figures are written under
`data/figures/`.

Every script here reads released artifacts rather than retraining, so unzip the required
directories from Zenodo first — see [Data and checkpoints](data.md). The **Requires** line on
each script tells you which one.

## Performance plots

These read the benchmark result JSONs in `data/final_benchmarks/` (and, for the PPO
ablation, `data/final_benchmarks_extra/`).

### Radar plot

Radar chart comparing multi-task performance (PPO, BC, Arnold) against the single-task
experts.

- **Script**: `plotting/plot_radar.py`
- **Requires**: `data/final_benchmarks/`
- **Output**: `data/figures/radar_plot_ppo_bc_arnold.png` / `.svg`

```bash
python plotting/plot_radar.py
```

### PPO ablation bar plot

Per-task bar plot comparing PPO variants (PPO, w/o reward norm, w/o observation norm,
MT-PPO) against the experts.

- **Script**: `plotting/plot_ppo_ablation_bars.py`
- **Requires**: `data/final_benchmarks/`, `data/final_benchmarks_extra/`
- **Output**: `data/figures/bar_plot_ppo_ablation.png` / `.svg`

```bash
python plotting/plot_ppo_ablation_bars.py
```

| Bar | Source directory | Seeds |
| --- | --- | --- |
| PPO | `data/final_benchmarks/ppo_t_sv/` | 3 |
| PPO w/o rew norm | `data/final_benchmarks_extra/ppo_wo_rew_norm/` | 1 |
| PPO w/o obs norm | `data/final_benchmarks/ppo_t/` | 3 |
| MT-PPO | `data/final_benchmarks/mt_ppo/` | 3 |

The PPO w/o reward norm arm is a single run rather than a 3-seed average, so it carries no
error bar. See [Multi-task RL baselines](mt-baselines.md) for the MT-PPO bar.

### Arnold ablation bar plot

Per-task bar plot comparing agent variants (BC, OBC w/o obs norm, OBC, OBC-PPO, Arnold)
against the experts, plus an improvement-over-baseline version.

- **Script**: `plotting/plot_arnold_ablation_bars.py`
- **Requires**: `data/final_benchmarks/`
- **Output**:
    - `data/figures/bar_plot_arnold_ablation.png` / `.svg`
    - `data/figures/bar_plot_arnold_ablation_improvement.png` / `.svg`

```bash
python plotting/plot_arnold_ablation_bars.py
```

## Learning curves

These read raw TensorBoard logs shipped alongside the checkpoints, or cached CSVs.

### RL fine-tuning curves

Compares the base multi-task OBC policy against several single-task policies fine-tuned
with PPO, plotting the solved fraction versus training steps from TensorBoard logs.
Experiment paths are set near the top of the script.

- **Script**: `plotting/plot_rl_finetuning_curves.py`
- **Requires**: `data/expert_policies/`, `data/student_policies/` (TensorBoard logs)
- **Output**: `data/figures/rl_finetuning_combined/rl_finetuning_combined_solved_curves.png` / `.svg`

```bash
python plotting/plot_rl_finetuning_curves.py
```

### Multi-task RL baselines (MT-SAC vs. MT-PPO)

Plots multi-task RL baseline learning curves comparing MT-SAC and MT-PPO across all tasks.

- **Script**: `plotting/plot_mt_algos.py`
- **Requires**: `data/final_benchmarks_extra/mt-curves/` (cached CSVs)
- **Output**: `data/figures/mt_algos_training_curves.png` / `.svg`

```bash
python plotting/plot_mt_algos.py
```

!!! note
    This script runs offline from the cached CSVs. Missing curves are re-fetched from
    Weights & Biases and re-cached, which requires access to the original runs — see
    [Data and checkpoints](data.md#weights-biases).

### Single-task student policy curves

Plots the learning curves of single-task student policies (PPO fine-tuning) after the
multi-task OBC student curves. Relies on the raw TensorBoard frames.

- **Script**: `plotting/plot_student_policy_curves.py`
- **Requires**: `data/student_policies/` (TensorBoard logs)
- **Output**: `data/figures/student_policies/` (`.png`)

```bash
python plotting/plot_student_policy_curves.py
```

### Transfer vs. from-scratch

Compares learning from a pretrained multi-task policy (Transfer) against training from
scratch for four downstream tasks: `pen`, `reorient`, `hand_middle_reach` and
`hand_little_reach`.

- **Script**: `plotting/plot_transfer_vs_scratch.py`
- **Requires**: `data/student_policies/` (TensorBoard logs)
- **Output**: `data/figures/transfer_learning/transfer_vs_scratch_comparison.png` / `.svg`

```bash
python plotting/plot_transfer_vs_scratch.py
```
