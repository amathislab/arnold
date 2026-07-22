# Arnold: a multi-task, multi-embodiment muscle transformer policy

## Model checkpoints and benchmark results
Available on [Zenodo](https://zenodo.org/records/21493316?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjYyZjg2NzFmLTFjNDUtNDAyZS04ZGY3LWVkOWY2OWE2ODM0OCIsImRhdGEiOnt9LCJyYW5kb20iOiIwNzc1MDkwYjMxYTY4MWM5YjQwZjk1OTQwNTgwZmVjOSJ9.T4Qw3-t9hmBYO4T8q1ofYsMJTOy9EejqbSkZ3WF2Iigf0_Ro8oKm1qi6WmAVEl9H7Lz-uVyRZIO8w2UXy8hElA)

What is contained?

1) We provide the code and the scripts to train policies with BC, PPO, OBC, OBC-PPO, RL fine-tuning and self-distillation. We thus also provide expert policies for imitation learning.
2) We also include pretrained checkpoints for OBC and OBC-PPO, to reproduce results and videos.

## Installation

To reproduce the training experiments, we are providing you two ways to set up a feasible environment.

The installation should usually take less than 30 minutes on a modern computer with fast internet connection.

### Method 1: Docker

Use the provided Dockerfile, you can create a docker container that can be then used to run all the arnold experiments (note: this assume that Docker is installed in your system).

To build the Docker image, navigate to root directory of this project containing the `Dockerfile` (docker-cuda) and run:

```bash
docker build -t arnold_image .
```

Once the image is built, you can run a container with:

```bash
docker run -it --rm arnold_image /bin/bash
```

This will start an interactive session within the container, where you can then execute the training or evaluation scripts.

### Method 2: Conda environment

You can create a conda environment and manually install all the dependencies.

```bash
conda create -n arnold python=3.8
conda activate arnold
pip install \
    cloudpickle==1.2.2\
    gym==0.13.0\
    gymnasium==0.29.1\
    h5py==3.7.0\
    wandb\
    tqdm\
    numpy\
    ipdb

pip install stable-baselines3==2.2.1
pip install MyoSuite==2.2.0
pip install imitation==1.0.0
pip install sb3-contrib==2.2.1
pip install Shimmy==1.3.0
pip install imageio
```

You may need to install some opengl-related system packages:

```bash
apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6
```

## Downloading Pretrained Models and Expert Policies

~~ The pretrained student policies and expert policies are not included in this repository due to their size. After downloading, unzip the `student_policies` and `expert_policies` folders and place them directly into the `data/` directory of this repository. Your `data/` directory should then contain subdirectories like `data/student_policies/` and `data/expert_policies/`.~~

For the reviewers of the paper, we are sharing a zipped folder that contains both the code and the weights. If you are reading this README then you already have access to the weights and code! 

## Arnold Training

To start training experiments, execute `src/main_train.py`. The parameters and checkpoint will be stored at `output/training/ongoing`. Here are example commands for two primary training approaches:

**1. OBC from scratch:**
This command starts an On-policy Behavioral Cloning (OBC) training from scratch using all the 14 tasks.
It assumes expert demonstrations are available when `imitation_coef > 0`.

```bash
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
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

**2. Final Arnold agent (BC imitating super-experts):**
This command trains the final Arnold agent by imitating specified super-expert policies, resuming from a pre-trained OBC model.

```bash
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
    --load_path data/student_policies/obc \
    --num_envs_per_task 2 \
    --ent_coef=0 \
    --vf_coef=0.5 \
    --pg_coef=0 \
    --imitation_coef=1 \
    --num_steps=2000000 \
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

Here's how to configure `src/main_train.py` for these and other training types in more detail:

- **PPO**: Set `--imitation_coef=0`, `--pg_coef=1`, `--ent_coef=1e-6`, and `--lr=2e-5`.
- **OBC-PPO**: Similar to PPO, but set `--imitation_coef=1` and `--lr=1e-4`.
- **BC**: This involves training with `imitation_coef > 0` and `pg_coef = 0`. The 'Final Arnold agent' command above is an example. If performing online imitation of an expert policy, the `--use_expert_actions` flag is typically used.
- **Super-experts**: Same as PPO, but resume from an OBC training (note: the reference to "second OBC example below" in the original text may refer to evaluation model examples; ensure you are loading a relevant trained OBC model checkpoint). Use a low learning rate (`--lr=2e-6`) and 32 instances of the same task (e.g., `--num_envs_per_task=32 --tasks <single_task_name>`).

## Evaluation - pretrained Models (OBC, Arnold, Experts)

The `src/benchmark.py` script allows you to evaluate the performance of various pretrained models, including those trained with OBC, Arnold, and expert policies.

### 1. Evaluating OBC and Arnold Models

To test a model trained with OBC or Arnold, you need to specify the path to the saved model (`.zip` file) and the task you want to evaluate. We provide trained models that you can download from [Here](https://drive.google.com/drive/folders/1V0xFcUtpfkUid-H-8w1EIx0ipxWK122H?usp=share_link).

Here's an example command:

```bash
python src/benchmark.py \
    --load path/to/your/model.zip \
    --task <task_names> \
    --arnold \
    --num_episodes <number_of_episodes> \
    --deterministic \
    --device <cpu_or_cuda> \
    --render
```

- Replace `path/to/your/model.zip` with the actual path to your trained model.
- Set `<task_names>` to one or many of the available tasks.
- Include the `--arnold` flag if the model was trained with Arnold.
- Adjust `--num_episodes` to set how many episodes to run for evaluation.
- Use `--deterministic` for deterministic actions from the policy.
- Specify the `--device` (e.g., `cpu` or `cuda`).
- Optionally render to video with `--render` (important: on Mac this requires running `mjpython` instead of `python`)

**Example for an Arnold model:**

```bash
python src/benchmark.py \
    --load data/student_policies/arnold/rl_model_64670238_steps.zip \
    --task kinesis \
    --arnold \
    --num_episodes 10 \
    --deterministic \
    --device cpu
```

**Example for a OBC model:**

```bash
python src/benchmark.py \
    --load data/student_policies/obc/rl_model_54974700_steps.zip \
    --task relocate \
    --arnold \
    --num_episodes 10 \
    --deterministic \
    --device cpu
```

Usually to reproduce these evaluations, the runtime is less than 1 minute.

### 2. Evaluating Expert Policies

To evaluate an expert policy, you need to specify the task.

```bash
python src/benchmark.py \
    --task <task_name> \
    --expert \
    --num_episodes <number_of_episodes> \
    --deterministic \
    --device <cpu_or_cuda>
```

- Set `<task_name>` to one of the available tasks.
- Include the `--expert` flag to indicate you are testing an expert policy.

**Example for an expert policy:**

```bash
python src/benchmark.py \
    --task relocate \
    --expert \
    --num_episodes 10 \
    --deterministic \
    --device cpu \
    --render
```

### 3. Available Tasks for Evaluation

You can choose `<task_name>` from the following list:

- `baoding_p1_ccw`
- `baoding_p1_cw`
- `baoding_p2`
- `baoding_p2_overlap`
- `hand_thumb_reach`
- `hand_index_reach`
- `hand_middle_reach`
- `hand_ring_reach`
- `hand_little_reach`
- `pen`
- `relocate`
- `reorient`
- `elbow_pose`
- `kinesis`

## Generating Performance Plots (used in the paper)

This section describes how to generate various performance plots using the provided scripts. The plots are typically saved in the `data/figures/` directory.

### Radar Plot

- **Script**: `src/plot_radar.py`
- **Description**: This script generates a radar chart, which is useful for visualizing the multi-task performance of different agents across a range of tasks. It helps in comparing the capabilities of agents in a comprehensive manner.
- **Output**: The generated radar plot is saved in the `data/figures/` directory (e.g., `data/figures/radar_plot.png`).

### PPO Ablation Bar Plot

- **Script**: `src/plot_ppo_ablation_bars.py`
- **Description**: This script creates bar plots to compare the performance of different Proximal Policy Optimization (PPO) algorithm configurations. Specifically, it visualizes ablations such as PPO with and without reward normalization, and PPO with and without observation normalization, comparing their performance against expert policies across various tasks.
- **Output**: The plots are saved as `data/figures/bar_plot_ppo_ablation.png` and `data/figures/bar_plot_ppo_ablation.svg`.

### Arnold Ablation Bar Plot

- **Script**: `src/plot_arnold_ablation_bars.py`
- **Description**: This script generates bar plots for an ablation study on the Arnold agent. It compares different versions of the agent (e.g., Behavioral Cloning (BC), On-policy Behavioral Cloning (OBC), OBC-PPO, and the full Arnold model) against expert policies. The script can also generate plots showing the performance improvement of these configurations over a baseline.
- **Output**: The output figures are saved in the `data/figures/` directory. Filenames typically include `bar_plot_arnold_ablation.png`/`.svg` for direct performance comparison and `bar_plot_arnold_ablation_improvement.png`/`.svg` for plots showing performance improvement.

## PCA Analysis of Trained Policies

This section outlines the steps to perform Principal Component Analysis (PCA) on the action space of trained policies. This analysis helps understand the effective dimensionality of the policy's actions and its impact on performance.

### 1. Collecting Activations and Actions

- **Script**: `src/collect_activations.py`
- **Description**: This script runs a trained policy in its environment for a specified number of episodes. It collects various data points, including observations, action means, rewards, and, for transformer-based policies like Arnold, intermediate layer activations. The collected action means are essential for the subsequent PCA analysis.
- **Output**: The script saves the data for each episode in HDF5 (`.h5`) files. These files are typically stored in a subdirectory within `data/activations/`, often named using the policy's identifier and checkpoint number (e.g., `data/activations/285_64670238/`).
- **Example Usage**:

  ```bash
  # Example for collecting activations for multiple tasks
  for task in hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach reorient pen baoding_p1_ccw baoding_p1_cw baoding_p2 baoding_p2_overlap; do
      python src/collect_activations.py \
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

- **Script**: `src/analyze_pca_inactivation.py`
- **Description**: This script uses the action data collected in the previous step. It performs PCA on these actions (either on a per-task basis or globally across all tasks). Then, it evaluates the policy's performance while progressively "inactivating" principal components. This involves projecting the policy's actions onto a lower-dimensional subspace defined by a decreasing number of the most significant principal components. This analysis helps to understand how performance is affected as the dimensionality of the allowed action space is reduced.
- **Output**: The script saves its analysis results, which include performance metrics for different numbers of active components, as pickle (`.pkl`) files. These are typically stored in a subdirectory within `data/pca_analysis/` (e.g., `data/pca_analysis/285_64670238/`).
- **Example Usage**:

  ```bash
  python src/analyze_pca_inactivation.py
  ```

  *(Note: The `analyze_pca_inactivation.py` script may contain hardcoded paths for input data and output directories. You might need to adjust these paths within the script to align with your specific data locations and desired output structure.)*

### 3. Plotting PCA Inactivation Performance

- **Script**: `src/plot_pca_inactivation.py`
- **Description**: This script takes the results from the PCA inactivation analysis (step 2) and generates plots. These plots typically illustrate the policy's performance (e.g., normalized solved steps or reward) as a function of the number of principal components used to define the action space. It can produce plots comparing performance when using task-specific PCAs versus a global PCA derived from actions across all tasks.
- **Output**: The generated plots are saved as image files (e.g., `.png`, `.svg`) in a subdirectory, usually within `data/figures/pca_inactivation/` (e.g., `data/figures/pca_inactivation/285_64670238/`).
- **Example Usage**:
  *(This script is intended to be run after `analyze_pca_inactivation.py` has generated its output files. It reads data from the output directory of the analysis script.)*

  ```bash
  python src/plot_pca_inactivation.py
  ```

### 4. Plotting Cumulative Explained Variance of Actions

- **Script**: `src/plot_action_pca_variance.py`
- **Description**: This script also utilizes the action data collected in step 1. It performs PCA on the action data (both per-task and globally) and then plots the cumulative explained variance as a function of the number of principal components. These plots help visualize how many dimensions are required to capture a certain percentage of the variance in the policy's action outputs, providing insights into the effective dimensionality of the learned action representations.
- **Output**: The plots are saved as image files (e.g., `.png`, `.svg`) in a subdirectory, typically within `data/figures/cumulative_variance/` (e.g., `data/figures/cumulative_variance/285_64670238/`).
- **Example Usage**:

  ```bash
  python src/plot_action_pca_variance.py --activations_dir data/activations/285_64670238 --out_dir data/figures/cumulative_variance/285_64670238
  ```

### 5. CSI Analysis

- **Script**: `src/plot_csi_analysis.py` and `src/plot_csi_curves.py`
- **Description**: Generates the figures for CSI analysis in the paper. `src/plot_csi_analysis.py` will measure the performance of a MLP policy that is constrained on a sub action space and fine-tuned with either RL or BC or not finetuned. `src/plot_csi_curves.py` will plot the learning curves for these experiments.


## Plotting RL Fine-tuning Learning Curves

- **Script**: `src/plot_rl_finetuning_curves.py`
- **Description**: This script generates learning curves that compare the performance of a base multi-task On-policy Behavioral Cloning (OBC) policy with several single-task policies fine-tuned using PPO. It reads training progress (specifically, the 'solved fraction' metric) from TensorBoard event files for both the base policy and the fine-tuned experiments. The script then plots these learning curves on a single graph, showing the solved fraction as a function of training steps. This visualization is useful for assessing the effectiveness of fine-tuning the generalist OBC policy on specific downstream tasks.
- **Input**: The script expects TensorBoard event files for the base OBC policy and for each of the RL fine-tuning experiments. The paths to these experiments and specific TensorBoard log directories are typically defined within the script itself (e.g., `BASE_EXPERIMENT_PATH`, `FINETUNE_EXPERIMENTS_BASE_DIR`, `FINETUNE_EXPERIMENT_NAMES`, and `TB_SUBDIR_CANDIDATES`). You may need to adjust these paths if your experiment data is stored elsewhere.
- **Output**: The script saves the combined learning curve plot as image files (e.g., `rl_finetuning_combined_solved_curves.png` and `.svg`) in the `data/figures/rl_finetuning_combined/` directory.
- **Example Usage**:
  *(Ensure that the paths to your TensorBoard logs are correctly configured within the script before running.)*

  ```bash
  python src/plot_rl_finetuning_curves.py
  ```

## License

This project is licensed under TBD. Keep it confidential until publication.

See the [LICENSE](LICENSE) file for details.
