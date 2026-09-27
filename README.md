# Arnold: a multi-task, multi-embodiment muscle transformer policy

Code for training and evaluating Arnold, OBC, PPO and BC policies.

## Installation

Run these commands from the repository root.

### Docker

```bash
docker build -t arnold_image docker-cuda
docker run -it --rm --gpus all \
    -v "$PWD":/arnold -w /arnold \
    arnold_image /bin/bash
```

For CPU use, omit `--gpus all` and pass `--device cpu` to experiment scripts.

### Conda

```bash
conda env create -f environment.yml
conda activate arnold
pip install imitation==1.0.0 cloudpickle==3.1.1
```

Install `imitation` last. On Linux, install the rendering libraries:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libosmesa6
```

## Download data and pretrained models

Download the [Zenodo v3 release](https://zenodo.org/records/21807280) from the
repository root using Python 3.8 or newer:

```bash
# Benchmark results (3.2 MB).
python scripts/fetch_data.py --profile benchmarks

# Model checkpoints, expert policies and Kinesis assets (3.4 GB).
python scripts/fetch_data.py --profile models

# Download both profiles.
python scripts/fetch_data.py --profile all
```

Files are installed under `data/`. See [Data and checkpoints](docs/data.md) for
paths and download options.

## Arnold Training

**1. OBC from scratch:**
This command starts an On-policy Behavioral Cloning (OBC) training from scratch using all the 14 tasks.
It assumes expert demonstrations are available when `imitation_coef > 0`.

```bash
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis \
        relocate baoding_p1_cw baoding_p2 baoding_p2_overlap kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
    --num_envs_per_task 2 \
    --ent_coef=0 \
    --vf_coef=0.5 \
    --pg_coef=0 \
    --imitation_coef=1 \
    --num_steps=50000000 \
    --batch_size=128 \
    --rollout_steps=512 \
    --embedding_size=128 \
    --dim_feedforward=512 \
    --num_heads=4 \
    --num_layers=6 \
    --lr=1e-3 \
    --log_interval=1 \
    --n_epochs=3 \
    --separate_vf_decoder \
    --policy_outputs_variance \
    --norm_reward \
    --dense_reward \
    --out_prefix=obc_ \
    --seed 1 \
    --project_name arnold_obc_multi_task_scratch
```

**2. OBC at reduced learning rate:**
After the first 50M steps, OBC (as well as BC and OBC-PPO) is continued for 5M additional steps at a smaller learning rate, which boosts the performance in certain tasks. This is the same command as above with `--load_path` set to the 50M checkpoint, `--num_steps=5000000` and `--lr=1e-5`. The resulting 55M-step checkpoint (`rl_model_54974700_steps.zip`) is the starting point for the RL fine-tuning and for the final Arnold agent.

**3. Final Arnold agent (OBC with super-experts):**
This command trains the final Arnold agent by imitating specified super-expert policies, resuming from the 55M-step OBC model.

```bash
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis \
        relocate baoding_p1_cw baoding_p2 baoding_p2_overlap kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
    --load_path data/final_checkpoints/obc/seed_0 \
    --num_envs_per_task 2 \
    --ent_coef=0 \
    --vf_coef=0.5 \
    --pg_coef=0 \
    --imitation_coef=1 \
    --num_steps=10000000 \
    --batch_size=128 \
    --rollout_steps=512 \
    --embedding_size=128 \
    --dim_feedforward=512 \
    --num_heads=4 \
    --num_layers=6 \
    --lr=1e-5 \
    --log_interval=1 \
    --n_epochs=3 \
    --separate_vf_decoder \
    --policy_outputs_variance \
    --norm_reward \
    --custom_experts data/expert_configs/arnold_experts_seed_1.json \
    --dense_reward \
    --out_prefix=285_ \
    --seed 1 \
    --project_name arnold_final_bc_super_experts
```

Here's how to configure `src/main_bc_ppo_multi_task.py` for these and other training types in more detail:

- **PPO**: Set `--imitation_coef=0`, `--pg_coef=1`, `--ent_coef=1e-6`, and `--lr=2e-5`.
- **OBC-PPO**: Similar to PPO, but set `--imitation_coef=1` and `--lr=1e-3` (reduced to `1e-5` for the additional 5M steps).
- **BC**: Add `--use_expert_actions` to the OBC command.
- **Super-experts**: Resume PPO from the 55M-step OBC checkpoint with `--lr 2e-6 --num_envs_per_task 32 --tasks <task> --reset_std --log_std_init -3`.

See [Training](docs/training.md) for atomic vocabulary and Task-SV options.

## Evaluation

Download the models with `python scripts/fetch_data.py --profile models`.
Run 200 stochastic episodes per task with a fixed evaluation seed:

```bash
python src/benchmark.py \
    --load data/final_checkpoints/arnold/seed_0/rl_model_64670238_steps.zip \
    --arnold --normalize --num_episodes 200 --seed 0 --device cpu \
    --save_results --out_dir data/final_benchmarks/arnold/seed_0
```

Tasks are read from the checkpoint's `args.json`. Use `--task <task_names>` to
select tasks. Atomic and Task-SV checkpoints use their saved vocabulary and
task embeddings automatically. Add `--deterministic` for deterministic actions.
`--normalize` restores the saved observation-normalization settings and freezes
the statistics, including for checkpoints trained with normalization disabled.

For MT-SAC and MT-PPO, use the vectorized benchmark:

```bash
python src/benchmark_multi_task_mlp.py \
    --load data/final_checkpoints/mt-ppo/seed_0/rl_model_60192776_steps.zip \
    --num_episodes 200 --seed 0 --device cpu \
    --save_results --out_dir data/final_benchmarks/mt_ppo/seed_0
```

Evaluate specialist teachers with deterministic actions:

```bash
python src/benchmark.py --task relocate --expert \
    --num_episodes 200 --seed 0 --device cpu \
    --save_results --out_dir data/final_benchmarks/expert_policies
```

Use `--expert_stochastic` to sample teacher actions, or
`--custom_experts data/expert_configs/arnold_experts_seed_1.json` for super-experts.
For rendering, add `--render`; on macOS use `mjpython`.

Results contain each episode's reward, length and solved-step count. The solved
fraction divides that count by the fixed task horizon. Performance plots divide
the mean fraction by the corresponding specialist teacher's mean fraction.

## Generating Performance Plots (used in the paper)

Scripts for the paper's performance plots. Figures are written under `data/figures/`.

### Radar Plot

- **Script**: `plotting/plot_radar.py`
- **Description**: Radar chart comparing multi-task performance (PPO, BC, Arnold) against the single-task experts.
- **Output**: `data/figures/radar_plot_ppo_bc_arnold.png` / `.svg`.
- **Example Usage**:

  ```bash
  python plotting/plot_radar.py
  ```

### PPO Ablation Bar Plot

- **Script**: `plotting/plot_ppo_ablation_bars.py`
- **Description**: Per-task bar plot comparing PPO variants (PPO, w/o reward norm, w/o observation norm, MT-PPO) against the experts.
- **Output**: `data/figures/bar_plot_ppo_ablation.png` / `.svg`.
- **Example Usage**:

  ```bash
  python plotting/plot_ppo_ablation_bars.py
  ```

### Arnold Ablation Bar Plot

- **Script**: `plotting/plot_arnold_ablation_bars.py`
- **Description**: Per-task bar plot comparing agent variants (BC, OBC, OBC-PPO, Arnold) against the experts, plus an improvement-over-baseline version.
- **Output**: `data/figures/bar_plot_arnold_ablation.png` / `.svg` and `data/figures/bar_plot_arnold_ablation_improvement.png` / `.svg`.
- **Example Usage**:

  ```bash
  python plotting/plot_arnold_ablation_bars.py
  ```

## PCA Analysis of Trained Policies

Principal Component Analysis (PCA) of the action space of trained policies, to study the effective dimensionality of the learned actions. Run the steps in order.

### 1. Collecting Activations and Actions

- **Script**: `plotting/collect_activations.py`
- **Description**: Rolls out a trained policy and saves per-episode observations, action means, rewards, and (for Arnold) intermediate activations. The action means feed the PCA steps below.
- **Output**: HDF5 (`.h5`) files under `data/activations/<policy_id>/` (e.g., `data/activations/285_64670238/`).
- **Example Usage**:

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

### 2. Running PCA Inactivation Analysis

- **Script**: `plotting/analyze_pca_inactivation.py`
- **Description**: Runs PCA on the collected actions (per-task and global) and measures performance while progressively inactivating principal components. Input/output paths are hardcoded near the top of the script.
- **Output**: Pickle (`.pkl`) files under `data/pca_analysis/<policy_id>/`.
- **Example Usage**:

  ```bash
  python plotting/analyze_pca_inactivation.py
  ```

### 3. Plotting PCA Inactivation Performance

- **Script**: `plotting/plot_pca_inactivation.py`
- **Description**: Plots performance (from step 2) versus the number of active principal components, comparing per-task and global PCA. Run after step 2.
- **Output**: `data/figures/pca_inactivation/<policy_id>/` (`.png` / `.svg`).
- **Example Usage**:

  ```bash
  python plotting/plot_pca_inactivation.py
  ```

### 4. Plotting Cumulative Explained Variance of Actions

- **Script**: `plotting/plot_action_pca_variance.py`
- **Description**: Plots the cumulative explained variance of the actions (per-task and global) versus the number of principal components.
- **Output**: `data/figures/cumulative_variance/<policy_id>/` (`.png` / `.svg`).
- **Example Usage**:

  ```bash
  python plotting/plot_action_pca_variance.py --activations_dir data/activations/285_64670238 --out_dir data/figures/cumulative_variance/285_64670238
  ```

## CSI Analysis

The released benchmark results in `data/final_benchmarks_extra/` already cover every arm, so **to reproduce only the figures, skip to step 5**. Steps 1-4 regenerate the experiments from scratch.

Below, `<task>` is one of the 14 tasks listed above and `<dim>` is one of the seven action-space sizes used in the paper: `1, 2, 5, 10, 20, 30, 40`.

### 1. Train the base MLP policy (OBC, 5M steps)

Trains the unconstrained MLP policy with On-policy Behavioral Cloning (`imitation_coef=1`, `pg_coef=0`).

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

### 2. Extract the CSI action subspace

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

### 3. Constrain to `<dim>` components and fine-tune

Three arms are compared, all constrained to the same subspace:

**a. Frozen (black)** — the policy is constrained but not trained further. Nothing to run here; it is evaluated directly from the step-1 checkpoint in step 4.

**b. OBC fine-tuned (red)** — 5M additional steps of On-policy Behavioral Cloning inside the subspace:

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

**c. PPO fine-tuned (blue)** — 5M additional steps of PPO inside the subspace. This is the same script with the imitation term switched off (`imitation_coef=0`, `pg_coef=1`). `--load_vecnormalize` restores the observation-normalization statistics saved next to the checkpoint, which RL fine-tuning requires:

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

Both fine-tuning arms resume from the 5M-step base checkpoint, so their own checkpoints are saved at `rl_model_10000000_steps.zip`.

### 4. Benchmark each arm

Evaluate every (task, `<dim>`) pair and write the results where the plotting scripts look for them:

| Arm | Checkpoint to evaluate | `--out_dir` |
| --- | --- | --- |
| Frozen (black) | step-1 base checkpoint (5M) | `data/final_benchmarks_extra/csi_notrain_server/csi<dim>_all` |
| OBC fine-tuned (red) | step-3b checkpoint (10M) | `data/final_benchmarks_extra/csi_bc_server/csi<dim>_all` |
| PPO fine-tuned (blue) | step-3c checkpoint (10M) | `data/final_benchmarks_extra/csi_server/csi<dim>_all` |

For the **frozen** arm the subspace is applied at evaluation time, so pass `--csi_components`:

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

For the **fine-tuned** arms the projection is already stored in the checkpoint, so `--csi_components` is omitted:

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

(On macOS, rendering requires `mjpython` instead of `python`.)

### 5. Generate the figures

- **Script**: `plotting/plot_csi_analysis.py` and `plotting/plot_csi_curves.py`
- **Description**: `plot_csi_analysis.py` plots final performance of the frozen and fine-tuned policies as a function of action-space size; `plot_csi_curves.py` plots the corresponding fine-tuning learning curves.
- **Output**: `data/figures/csi_analysis/` (`.png` / `.svg`).
- **Example Usage**:

  ```bash
  python plotting/plot_csi_analysis.py
  python plotting/plot_csi_curves.py
  ```

## Training and Benchmarking Multi-Task RL Baselines (MT-SAC vs. MT-PPO)

Both baselines are trained with the same script, `src/main_sac_multi_task.py` (multi-task SAC/PPO with an MLP policy); MT-PPO is selected with `--algo ppo`.

### 1. Training

**MT-SAC:**

```bash
python src/main_sac_multi_task.py \
    --project_name arnold-new-exp \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_ccw baoding_p1_cw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis \
    --num_envs_per_task 2 \
    --num_steps 50_000_000 \
    --hidden_size 512 \
    --num_layers 2 \
    --train_freq 64 \
    --save_freq 100000 \
    --norm_reward \
    --lr 1e-5 \
    --out_prefix sac-run- \
    --seed 771
```

**MT-PPO** — the same script with `--algo ppo` and a rollout length:

```bash
python src/main_sac_multi_task.py \
    --project_name arnold-mtppo-new \
    --algo ppo \
    --rollout_steps 256 \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_ccw baoding_p1_cw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis \
    --num_envs_per_task 2 \
    --num_steps 50_000_000 \
    --hidden_size 512 \
    --num_layers 2 \
    --train_freq 64 \
    --save_freq 100000 \
    --norm_reward \
    --lr 1e-5 \
    --log_interval 1 \
    --out_prefix sac-run- \
    --seed 771
```

### 2. Benchmarking

Both baselines are evaluated with `src/benchmark_multi_task_mlp.py`. The task order, environment id and algorithm are recovered from the `args.json` saved next to each checkpoint, so only the checkpoint path is needed. Evaluate the latest checkpoint of each seed:

```bash
python src/benchmark_multi_task_mlp.py \
    --load <run_dir>/rl_model_<steps>_steps.zip \
    --num_episodes 200 \
    --device cpu \
    --save_results \
    --out_dir data/final_benchmarks/mt_ppo/seed_0
```

### 3. Figures

- **Learning curves**: `plotting/plot_mt_algos.py` (see [Plotting Learning Curves](#plotting-learning-curves)).
- **MT-PPO bar**: `plotting/plot_ppo_ablation_bars.py` aggregates `data/final_benchmarks/mt_ppo/` for the MT-PPO bar of the PPO ablation plot.

## Plotting Learning Curves

Learning-curve figures. Figures are written under `data/figures/`.

### RL Fine-tuning Curves

- **Script**: `plotting/plot_rl_finetuning_curves.py`
- **Description**: Compares the base multi-task OBC policy against several single-task policies fine-tuned with PPO, plotting the solved fraction versus training steps from TensorBoard logs. Experiment paths are set near the top of the script.
- **Output**: `data/figures/rl_finetuning_combined/rl_finetuning_combined_solved_curves.png` / `.svg`.
- **Example Usage**:

  ```bash
  python plotting/plot_rl_finetuning_curves.py
  ```

### Plotting Multi-Task RL Baselines (MT-SAC vs. MT-PPO)

- **Script**: `plotting/plot_mt_algos.py`
- **Description**: Plots multi-task RL baseline learning curves comparing MT-SAC and MT-PPO across all tasks.
- **Output**: `data/figures/mt_algos_training_curves.png` / `.svg`.
- **Example Usage**:

  ```bash
  python plotting/plot_mt_algos.py
  ```

### Single-Task Student Policy Curves

- **Script**: `plotting/plot_student_policy_curves.py`
- **Description**: Plots the learning curves of single-task student policies (PPO fine-tuning) after the multi-task OBC student curves. Relies on the raw TensorBoard frames.
- **Output**: `data/figures/student_policies/` (`.png`).
- **Example Usage**:

  ```bash
  python plotting/plot_student_policy_curves.py
  ```

### Transfer vs. From-Scratch

- **Script**: `plotting/plot_transfer_vs_scratch.py`
- **Description**: Compares learning from a pretrained multi-task policy (Transfer) against training from scratch for four downstream tasks (pen, reorient, hand_middle_reach, hand_little_reach).
- **Output**: `data/figures/transfer_learning/transfer_vs_scratch_comparison.png` / `.svg`.
- **Example Usage**:

  ```bash
  python plotting/plot_transfer_vs_scratch.py
  ```

## License

This project is licensed under the BSD 3-Clause License.

It also includes code derived from Stable-Baselines3, imitation, PyTorch-RL,
Lattice, Kinesis, PHC and IsaacGymEnvs, which remains under its own licenses
(MIT, BSD 3-Clause and BSD 3-Clause Clear).

The musculoskeletal models in `data/kinesis/xml/` are distributed by myo_sim and
Kinesis under the Apache License 2.0.

See the [LICENSE](LICENSE) file for the full terms and per-file attributions.
