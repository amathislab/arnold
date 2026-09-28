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
| PPO | `data/final_benchmarks/ppo_t_sv/` | 5 |
| PPO w/o rew norm | `data/final_benchmarks_extra/ppo_wo_rew_norm/` | 1 |
| PPO w/o obs norm | `data/final_benchmarks_extra/ppo_wo_obs_norm/` | 1 |
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
- **Requires**: `data/final_benchmarks_extra/rl_finetuning/` and `data/final_benchmarks/arnold_multi_task/` (TensorBoard logs)
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
- **Requires**: `data/final_benchmarks/arnold_single_task/` (TensorBoard logs)
- **Output**: `data/figures/student_policies/` (`.png`)

```bash
python plotting/plot_student_policy_curves.py
```

### Transfer vs. from-scratch

Compares learning from a pretrained multi-task policy (Transfer) against training from
scratch for four downstream tasks: `pen`, `reorient`, `hand_middle_reach` and
`hand_little_reach`.

- **Script**: `plotting/plot_transfer_vs_scratch.py`
- **Requires**: `data/final_benchmarks/arnold_multi_task/` (TensorBoard logs)
- **Output**: `data/figures/transfer_learning/transfer_vs_scratch_comparison.png` / `.svg`

```bash
python plotting/plot_transfer_vs_scratch.py
```

## Additional tables and plots

The exact benchmark selections are recorded in `data/analysis/reproduction/policies.json`;
shared task labels live in `src/analysis/metadata.py`.
These use the existing `data/final_benchmarks/` and `data/final_benchmarks_extra/` layout:

```bash
python plotting/ablation_table.py
python plotting/plot_relative_dotplot.py
python plotting/plot_capacity_performance.py
python plotting/plot_bilateral_reward.py
python plotting/plot_csi_analysis.py
```

Tables (CSV and LaTeX) are written under `data/analysis/tables/`. They report relative
solved-step fractions and SEM across seeds, with task-paired Wilcoxon tests and Holm
correction. The single-run capacity comparison uses episode SEM, explicitly reported in
its CSV. PPO normalization-ablation arms have one run and show no seed error bar. Radar
and PPO bars now consistently use the paper's relative solved-step fraction.

The PR's optional portable CSV plotting workflow supplements the existing TensorBoard
commands above:

```bash
python plotting/plot_learning_curves.py --panel finetuning --raw \
    --smoothing savgol --window 101 --combine_tasks
python src/analysis/export_learning_curves.py \
    --events path/to/events.out.tfevents.example --tag MuscleDieReorientP0-v0/solved \
    --panel transfer --method Transfer --task reorient --stage transfer --seed 0 \
    --output data/analysis/curves/transfer_reorient.csv
python plotting/plot_learning_curves.py \
    --curves data/analysis/curves/transfer_reorient.csv --panel transfer
```

Only the `finetuning` panel is included in `data/analysis/learning_curves.csv.gz`.
Other panels require exported CSVs. Columns are
`panel,method,seed,task,stage,step,value,metric`. Supply resumed event files in chronological
resume order. `--offsets stage=50000000` subtracts that value from that stage's recorded
steps. Multiple seeds are aligned on shared steps and shaded with standard deviation.
Portable plots go to `data/figures/learning_curves/`; the original learning-curve entry
points and their output paths are retained.

## Migrated training logs

The local single-task and multi-task folders were copied from the legacy repository,
including checkpoints, configuration and TensorBoard logs. The legacy copies are retained.
CSI event files and run arguments are stored under `training/` in `csi`,
`csi_server`, `csi_bc_server` and `csi_notrain_server`, within
`data/final_benchmarks_extra/`. The CSI curve plot keeps its selected runs: baseline 111,
RL 555 and OBC 666; other migrated runs remain available.

Arnold RL fine-tuning is a separate experiment, stored under `rl_finetuning/` in the
same parent folder. Its elbow curve uses the available `elbow_263_78374700` run,
whose arguments identify elbow pose and the 249 OBC base checkpoint; the previously
referenced `elbow_pose_271_95774700` run was unavailable. This is a different run,
so the curve is not claimed to reproduce that missing run.
