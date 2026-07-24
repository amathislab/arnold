# CSI-Finetuning

How well a policy performs when its action space is **constrained** to a low-dimensional
subspace, with and without further training inside that subspace. Corresponds to Figures
S11 and S14 of the paper.

!!! note "CSI-Finetuning vs. CSI analysis"
    This page *constrains* a policy's action space to a fixed subspace and then trains
    inside it. That is a different experiment from the [CSI analysis](csi-analysis.md),
    which projects the actions of an already-trained policy after the fact and measures the
    performance that survives.

!!! tip "Only want the figures?"
    The released benchmark results in `data/final_benchmarks_extra/` already cover every
    arm, so **to reproduce only the figures, skip to
    [Related figures](#related-figures)**. Steps 1–4 regenerate the experiments from
    scratch.

Throughout, `<task>` is one of the [14 tasks](tasks.md) and `<dim>` is one of the seven
action-space sizes used in the paper: `1, 2, 5, 10, 20, 30, 40`. Unlike the
[CSI analysis](csi-analysis.md), this experiment is not restricted to MyoHand — the released
results cover all 14 tasks at every dimension.

## 1. Train the base MLP policy (OBC, 5M steps)

Trains the unconstrained MLP policy with On-policy Behavioral Cloning (`imitation_coef=1`,
`pg_coef=0`).

```bash
python src/main_bc_ppo.py \
    --task <task> \
    --network csi \
    --num_envs 16 \
    --imitation_coef 1.0 \
    --pg_coef 0 \
    --vf_coef 0 \
    --num_steps 5000000 \
    --local
```

The checkpoint is written to `output/training/ongoing/<run_name>/rl_model_5000000_steps.zip`.

## 2. Extract the CSI action subspace

Rolls the trained policy out and runs PCA on its actions to obtain the control subspace.

```bash
python src/main_csi_get_subspace.py \
    --policy_path output/training/ongoing/<run_name>/rl_model_5000000_steps.zip \
    --task <task> \
    --num_envs 4 \
    --num_steps 100000 \
    --save_path output/<task>_csi
```

This writes `output/<task>_csi/subspace.npy` and `output/<task>_csi/mean.npy`.

## 3. Constrain to `<dim>` components and fine-tune

Three arms are compared, all constrained to the same subspace.

=== "a. Frozen (black)"

    The policy is constrained but not trained further. Nothing to run here — it is
    evaluated directly from the step-1 checkpoint in step 4.

=== "b. OBC fine-tuned (red)"

    5M additional steps of On-policy Behavioral Cloning inside the subspace:

    ```bash
    python src/main_bc_ppo.py \
        --task <task> \
        --network csi \
        --num_envs 16 \
        --imitation_coef 1.0 \
        --pg_coef 0.0 \
        --vf_coef 0.5 \
        --ent_coef 0.0 \
        --load_path output/training/ongoing/<run_name>/rl_model_5000000_steps.zip \
        --load_csi_subspace output/<task>_csi/subspace.npy \
        --csi_subspace <dim> \
        --num_steps 5000000 \
        --out_prefix 666_<task>_csi<dim>_
    ```

=== "c. PPO fine-tuned (blue)"

    5M additional steps of PPO inside the subspace. Same script with the imitation term
    switched off (`imitation_coef=0`, `pg_coef=1`). `--load_vecnormalize` restores the
    observation-normalization statistics saved next to the checkpoint, which RL fine-tuning
    requires:

    ```bash
    python src/main_bc_ppo.py \
        --task <task> \
        --network csi \
        --num_envs 16 \
        --imitation_coef 0 \
        --pg_coef 1 \
        --vf_coef 0.8 \
        --ent_coef 1e-6 \
        --load_path output/training/ongoing/<run_name>/rl_model_5000000_steps.zip \
        --load_vecnormalize \
        --load_csi_subspace output/<task>_csi/subspace.npy \
        --csi_subspace <dim> \
        --num_steps 5000000 \
        --out_prefix 333_<task>_csi<dim>_
    ```

Both fine-tuning arms resume from the 5M-step base checkpoint, so their own checkpoints are
saved at `rl_model_10000000_steps.zip`.

## 4. Benchmark each arm

Evaluate every (task, `<dim>`) pair and write the results where the plotting scripts look
for them:

| Arm | Checkpoint to evaluate | `--out_dir` |
| --- | --- | --- |
| Frozen (black) | step-1 base checkpoint (5M) | `data/final_benchmarks_extra/csi_notrain_server/csi<dim>_all` |
| OBC fine-tuned (red) | step-3b checkpoint (10M) | `data/final_benchmarks_extra/csi_bc_server/csi<dim>_all` |
| PPO fine-tuned (blue) | step-3c checkpoint (10M) | `data/final_benchmarks_extra/csi_server/csi<dim>_all` |

For the **frozen** arm the subspace is applied at evaluation time, so pass
`--csi_components`:

```bash
python src/benchmark.py \
    --policy csi \
    --load output/training/ongoing/<run_name>/rl_model_5000000_steps.zip \
    --task <task> \
    --csi_components output/<task>_csi/subspace.npy \
    --csi_subspace <dim> \
    --normalize --deterministic \
    --num_episodes 200 \
    --device cpu \
    --save_results \
    --out_dir data/final_benchmarks_extra/csi_notrain_server/csi<dim>_all
```

For the **fine-tuned** arms the projection is already stored in the checkpoint, so
`--csi_components` is omitted:

```bash
python src/benchmark.py \
    --policy csi \
    --load output/training/ongoing/<finetuned_run>/rl_model_10000000_steps.zip \
    --task <task> \
    --csi_subspace <dim> \
    --normalize --deterministic \
    --num_episodes 200 \
    --device cpu \
    --save_results \
    --out_dir data/final_benchmarks_extra/csi_bc_server/csi<dim>_all
```

!!! warning "macOS"
    Rendering requires `mjpython` instead of `python`.

## Related figures

| Figure | Paper | Script | Output |
| --- | --- | --- | --- |
| Final performance vs. action-space size, frozen and fine-tuned | S11 | `plotting/plot_csi_analysis.py` | `data/figures/csi_analysis/` (`.png` / `.svg`) |
| Fine-tuning learning curves for each arm | S14 | `plotting/plot_csi_curves.py` | `data/figures/csi_analysis/` (`.png` / `.svg`) |

!!! note "Script and output names"
    Both scripts and their output directory are named `csi_analysis`, but they belong to
    CSI-Finetuning, not to the [CSI analysis](csi-analysis.md) page.

Both read the benchmark result JSONs written in step 4, which ship in
`data/final_benchmarks_extra/` — so these run without steps 1–4.

```bash
python plotting/plot_csi_analysis.py
python plotting/plot_csi_curves.py
```
