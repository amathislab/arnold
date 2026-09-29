# CSI analysis

**Control subspace inactivation (CSI)** — the paper's muscle-synergy analysis. Muscle
activations produced by a trained policy are collected, Principal Component Analysis (PCA)
is run on them, and the control signal is then projected onto the subspace spanned by the
`N` most important principal components while task performance is measured. The result is a
*functional* measure of how many synergies the policy actually needs.

PCA is run two ways, and comparing them is the point of the analysis:

- **Per task** — principal components computed from a single task's activations.
- **Pooled** — principal components computed from all tasks' activations combined.

If the two curves coincide, the policy's synergies transfer across tasks; if the pooled
subspace needs many more components to reach the same performance, the low-dimensional
structure is task-specific.

!!! note "CSI analysis vs. CSI-Finetuning"
    This page projects the actions of an *already-trained* policy after the fact. It is a
    different experiment from [CSI-Finetuning](csi-finetuning.md), where a policy's action
    space is *constrained* to a fixed subspace and then trained inside it.

## Why only MyoHand tasks

The analysis runs over the **11 MyoHand tasks** — the five finger reaches, `pen`, `reorient`,
and the four Baoding variants. Only these share the same embodiment, and therefore the same
39-muscle action space, so principal components are comparable across them and can be pooled
into a shared subspace.

`elbow_pose` (MyoElbow, 6 muscles), `relocate` (MyoArm, 63) and `kinesis` (MyoLeg, 80) have
different action dimensionalities and are excluded. See [Tasks](tasks.md).

!!! warning "Run the steps in order"
    Each step consumes the output of the previous one.

## 1. Collecting activations and actions

Rolls out a trained policy and saves per-episode observations, action means, rewards, and
(for Arnold) intermediate activations. The action means feed the PCA steps below.

- **Script**: `plotting/collect_activations.py`
- **Output**: HDF5 (`.h5`) files under `data/activations/<policy_id>/`, e.g.
  `data/activations/example_64670238/`

```bash
# Collect activations for multiple tasks
for task in hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach reorient pen baoding_p1_ccw baoding_p1_cw baoding_p2 baoding_p2_overlap; do
    python plotting/collect_activations.py \
        --load data/final_benchmarks/example_checkpoint/rl_model_64670238_steps.zip \
        --task $task \
        --num_episodes 100 \
        --arnold \
        --normalize \
        --device cpu \
        --out_dir data/activations
done
```

## 2. Running PCA or NMF inactivation analysis

Supply the checkpoint and the directory containing either per-episode recordings or
compact signals:

```bash
python plotting/analyze_pca_inactivation.py \
    --load path/to/rl_model_64670238_steps.zip \
    --signals data/activations/example_64670238 \
    --method pca --scope task --num_episodes 100 \
    --out_dir data/pca_analysis/example_64670238/task
python plotting/analyze_pca_inactivation.py \
    --load path/to/rl_model_64670238_steps.zip \
    --signals data/activations/example_64670238 \
    --method pca --scope global --num_episodes 100 \
    --out_dir data/pca_analysis/example_64670238/global
```

Each run writes `curves.csv`, episode outcomes and fitted components. PCA centers the
recorded action means, then projects stochastic policy actions through the fitted basis.
The same global fit is reused across tasks. Performance divides solved steps by the fixed
task horizon, including episodes that terminate early.

For the additional NMF analysis, collect physical signals with
[collect_signals.py](signals.md), then use `--method nmf --n_fits 10`. NMF projects physical
actuator controls after actuator processing; it does not factor signed policy actions.
NMF uses ranks 1–38 and ten seeded fits; PCA uses ranks 1–39. `--dimensions` selects a subset.

## 3. Plotting PCA inactivation performance

```bash
python plotting/plot_pca_inactivation.py \
    --curves data/pca_analysis/example_64670238/task/curves.csv \
             data/pca_analysis/example_64670238/global/curves.csv
```

Without `--curves`, the command plots the included historical summary at
`data/analysis/historical_curves.csv`. That cache contains 20 episodes per rank for
task-specific PCA, 100 for global PCA, and 100 across ten NMF fits. Fresh runs default to
100 episodes, matching the supplied manuscript's PCA methods. Cached results and fresh
reproduction results should therefore be distinguished.

Figures are saved under `data/figures/pca_inactivation/` as PNG and SVG.

## 4. Plotting cumulative explained variance of actions

Plots the cumulative explained variance of the actions (per-task and global) versus the
number of principal components.

- **Script**: `plotting/plot_action_pca_variance.py`
- **Output**: `data/figures/cumulative_variance/<policy_id>/` (`.png` / `.svg`)

```bash
python plotting/plot_action_pca_variance.py \
    --activations_dir data/activations/example_64670238 \
    --out_dir data/figures/cumulative_variance/example_64670238
```

## Comparing subspaces

```bash
python plotting/analyze_subspaces.py --data_dir data/analysis/signals
python plotting/analyze_subspaces.py --selection capacity_recordings --successful 100
```

Selections are explicit in `data/analysis/reproduction/signals.json`; customize them with
`--selections`. Collect all listed policies/tasks first (see [Analysis signals](signals.md)).
The commands write pairwise measurements, summaries and figures to `data/figures/subspaces/`.
PVD is directed from the first condition to the second; PAD is the mean principal angle
in degrees. Shading is standard deviation across pairs. The default selections compare
Arnold, expert, multi-task OBC and single-task OBC; these are additional comparisons and
are not the multiple-Arnold-checkpoint comparison described in Figure 7 of the supplied PDF.


For learning curves of policies trained inside a constrained subspace, see
[CSI-Finetuning](csi-finetuning.md).
