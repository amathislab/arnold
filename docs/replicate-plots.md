# Replicate plots

Scripts for the paper's performance plots and learning curves. All figures are written under
`data/figures/`.

The plotting commands read evaluation results, TensorBoard logs or curve caches.
Install the inputs matching the current layout first — see [Data and checkpoints](data.md). The **Requires** line on
each script tells you which one.

## Performance plots

These read the benchmark result JSONs in `data/final_benchmarks/` (and, for the PPO
ablation, `data/final_benchmarks_extra/`).

Benchmark selections are recorded in `data/analysis/reproduction/policies.json`;
shared task labels live in `src/analysis/metadata.py`. Tables, relative-performance and
capacity plots read `data/final_benchmarks/`; bilateral rewards read
`data/final_benchmarks_extra/`. Radar and PPO bars use relative solved-step fractions.
PPO normalization-ablation arms have one run and show no seed error bar.

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

### Ablation Tables

```bash
python plotting/ablation_table.py
```

Writes CSV and LaTeX tables to `data/analysis/tables/`, reporting relative solved-step
fractions, SEM across seeds, and task-paired Wilcoxon tests with Holm correction.

### Relative-Performance Plot

```bash
python plotting/plot_relative_dotplot.py
```

Compares per-task solved-step fractions for `obc_task_sv` against `obc` by default.
Use `--method` and `--reference` to select policies. Writes `data/figures/relative_dotplot.svg`.

### Model Capacity Plot

```bash
python plotting/plot_capacity_performance.py
```

Compares OBC model sizes against expert performance. Writes `capacity.svg` and
`performance.csv` under `data/figures/capacity/`. Error bars show episode SEM for
one run per model size.

### Bilateral Reward Plot

```bash
python plotting/plot_bilateral_reward.py
```

Reads `data/final_benchmarks_extra/bilateral/bilateral.json` and compares episode
rewards. Writes `data/figures/bilateral_reward.svg`.

## Learning curves

These read raw TensorBoard logs shipped alongside the checkpoints, or cached CSVs.

### RL fine-tuning curves

Compares the base multi-task OBC policy against several single-task policies fine-tuned
with PPO, plotting the solved fraction versus training steps from TensorBoard logs.
Experiment paths are set near the top of the script.

- **Script**: `plotting/plot_rl_finetuning_curves.py`
- **Requires**: `data/final_benchmarks_extra/rl_finetuning/` and `data/final_benchmarks/example_training_curve/` (TensorBoard logs)
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

Plots single-task imitation-learning histories from per-task TensorBoard logs.

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
- **Requires**: `data/final_benchmarks/transfer_learning/` (TensorBoard logs)
- **Output**: `data/figures/transfer_learning/transfer_vs_scratch_comparison.png` / `.svg`

```bash
python plotting/plot_transfer_vs_scratch.py
```

### Learning curves from CSV inputs

Plot the supplied fine-tuning CSV cache, or export TensorBoard logs to CSV:

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
CSV-based plots go to `data/figures/learning_curves/`.

### Training-log inputs

Single-task, transfer and base OBC logs are under `data/final_benchmarks/` in
`arnold_single_task/`, `transfer_learning/` and `example_training_curve/`, respectively.
These folders contain training logs and configurations. The model and matching
normalization file used in examples are under `example_checkpoint/`.

Arnold RL fine-tuning logs are under `data/final_benchmarks_extra/rl_finetuning/`.
The elbow curve uses `elbow_263_78374700`, an elbow-pose run initialized from the
249 OBC base checkpoint. It does not reproduce the unavailable `elbow_pose_271_95774700` run.
