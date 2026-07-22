# Arnold: A multi-task, multi-embodiment muscle transformer policy

This repository provides the code to load, benchmark, and train Arnold — a generalist
muscle transformer policy — together with the multi-task musculoskeletal environments it
was trained on. It supports training with BC, PPO, OBC, OBC-PPO, RL fine-tuning, and
self-distillation, and evaluating pretrained student and expert policies.

## Model checkpoints and benchmark results
Available on [Zenodo](https://zenodo.org/records/21493316?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjYyZjg2NzFmLTFjNDUtNDAyZS04ZGY3LWVkOWY2OWE2ODM0OCIsImRhdGEiOnt9LCJyYW5kb20iOiIwNzc1MDkwYjMxYTY4MWM5YjQwZjk1OTQwNTgwZmVjOSJ9.T4Qw3-t9hmBYO4T8q1ofYsMJTOy9EejqbSkZ3WF2Iigf0_Ro8oKm1qi6WmAVEl9H7Lz-uVyRZIO8w2UXy8hElA)

After downloading, unzip the `student_policies` and `expert_policies` folders and place
them directly into the `data/` directory of this repository. Your `data/` directory should
then contain subdirectories like `data/student_policies/` and `data/expert_policies/`.

## Installation

We provide two ways to set up a working environment. Installation usually takes less than 30
minutes on a modern computer with a fast internet connection.

### Method 1: Docker

Use the provided Dockerfile to build a container that can run all Arnold experiments (this
assumes Docker is installed on your system). From the `docker-cuda/` directory containing the
`Dockerfile`, run:

```bash
docker build -t arnold_image .
```

Then start an interactive session:

```bash
docker run -it --rm arnold_image /bin/bash
```

### Method 2: Conda environment

Create a conda environment and install the dependencies manually:

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

You may need to install some OpenGL-related system packages:

```bash
apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6
```

## Load and benchmark a policy

The `src/benchmark.py` script evaluates the performance of pretrained models, including those
trained with OBC and Arnold, as well as expert policies. Download the checkpoints from the
[Zenodo archive](#model-checkpoints-and-benchmark-results) above and unzip them into `data/`.

### 1. Evaluating OBC and Arnold models

Specify the path to the saved model (`.zip`) and the task to evaluate:

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

- Replace `path/to/your/model.zip` with the path to your trained model.
- Set `<task_names>` to one or many of the available tasks (see the list below).
- Include `--arnold` if the model was trained with Arnold.
- Adjust `--num_episodes` to set how many episodes to run.
- Use `--deterministic` for deterministic actions from the policy.
- Specify `--device` (e.g. `cpu` or `cuda`).
- Optionally render to video with `--render` (on Mac this requires running `mjpython`
  instead of `python`).

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

**Example for an OBC model:**

```bash
python src/benchmark.py \
    --load data/student_policies/obc/rl_model_54974700_steps.zip \
    --task relocate \
    --arnold \
    --num_episodes 10 \
    --deterministic \
    --device cpu
```

Reproducing these evaluations usually takes less than 1 minute.

### 2. Evaluating expert policies

To evaluate an expert policy, specify the task and pass `--expert`:

```bash
python src/benchmark.py \
    --task <task_name> \
    --expert \
    --num_episodes <number_of_episodes> \
    --deterministic \
    --device <cpu_or_cuda> \
    --render
```

### 3. Available tasks

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

## Training

Training experiments are launched through `src/main_bc_ppo_multi_task.py`. Parameters and
checkpoints are stored under `output/training/ongoing`.

**1. OBC from scratch:**
This starts an On-policy Behavioral Cloning (OBC) training from scratch on all 14 tasks. It
assumes expert demonstrations are available when `imitation_coef > 0`.

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
This trains the final Arnold agent by imitating specified super-expert policies, resuming
from a pre-trained OBC model.

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

Other training regimes are configured through the same script:

- **PPO**: `--imitation_coef=0`, `--pg_coef=1`, `--ent_coef=1e-6`, `--lr=2e-5`.
- **OBC-PPO**: as PPO, but `--imitation_coef=1` and `--lr=1e-4`.
- **BC**: train with `imitation_coef > 0` and `pg_coef = 0` (the "Final Arnold agent"
  command above is an example). For online imitation of an expert policy, add
  `--use_expert_actions`.
- **Super-experts**: as PPO, resuming from an OBC training, with a low learning rate
  (`--lr=2e-6`) and 32 instances of a single task
  (e.g. `--num_envs_per_task=32 --tasks <single_task_name>`).

# Reference

```
@misc{chiappa2025arnoldgeneralistmuscletransformer,
      title={Arnold: a generalist muscle transformer policy}, 
      author={Alberto Silvio Chiappa and Boshi An and Merkourios Simos and Chengkun Li and Alexander Mathis},
      year={2025},
      eprint={2508.18066},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.18066}, 
}
```
