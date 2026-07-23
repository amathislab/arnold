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
  `data/activations/285_64670238/`

```bash
# Collect activations for multiple tasks
for task in hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach reorient pen baoding_p1_ccw baoding_p1_cw baoding_p2 baoding_p2_overlap; do
    python plotting/collect_activations.py \
        --load data/student_policies/arnold_multi_task/285_arnold_htr_hir_hmr_hrr_hlr_r_p_bpc_bpc_bp_bpo_ep_r_k_k_r_bpc_bp_bpo_k_k_r_bpc_bp_bpo_k_k_bc_ppo_seed_1/rl_model_64670238_steps.zip \
        --task $task \
        --num_episodes 100 \
        --arnold \
        --normalize \
        --device cpu \
        --out_dir data/activations
done
```

## 2. Running PCA inactivation analysis

Runs PCA on the collected actions (per-task and global) and measures performance while
progressively inactivating principal components.

- **Script**: `plotting/analyze_pca_inactivation.py`
- **Output**: Pickle (`.pkl`) files under `data/pca_analysis/<policy_id>/`

```bash
python plotting/analyze_pca_inactivation.py
```

!!! note "Hardcoded paths"
    Input and output paths are hardcoded near the top of the script. Edit them if your
    `<policy_id>` differs from the one used in step 1.

## 3. Plotting PCA inactivation performance

Plots performance (from step 2) versus the number of active principal components, comparing
per-task and global PCA. Run after step 2.

- **Script**: `plotting/plot_pca_inactivation.py`
- **Output**: `data/figures/pca_inactivation/<policy_id>/` (`.png` / `.svg`)

```bash
python plotting/plot_pca_inactivation.py
```

## 4. Plotting cumulative explained variance of actions

Plots the cumulative explained variance of the actions (per-task and global) versus the
number of principal components.

- **Script**: `plotting/plot_action_pca_variance.py`
- **Output**: `data/figures/cumulative_variance/<policy_id>/` (`.png` / `.svg`)

```bash
python plotting/plot_action_pca_variance.py \
    --activations_dir data/activations/285_64670238 \
    --out_dir data/figures/cumulative_variance/285_64670238
```

## Related figures

| Figure | Script | Output |
| --- | --- | --- |
| Performance vs. number of active principal components | [`plotting/plot_pca_inactivation.py`](#3-plotting-pca-inactivation-performance) | `data/figures/pca_inactivation/<policy_id>/` (`.png` / `.svg`) |
| Cumulative explained variance of the actions | [`plotting/plot_action_pca_variance.py`](#4-plotting-cumulative-explained-variance-of-actions) | `data/figures/cumulative_variance/<policy_id>/` (`.png` / `.svg`) |

Unlike the [CSI-Finetuning figures](csi-finetuning.md#related-figures), these have no
released intermediate results — steps 1 and 2 must be run first to produce the activations
and PCA pickles.
