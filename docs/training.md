# Training

Here are the list of training entry points of this project.

| Script | Use |
| --- | --- |
| `src/main_bc_ppo_multi_task.py` | Multi-task training — BC, PPO, OBC, OBC-PPO, RL fine-tuning. |
| `src/main_bc_ppo.py` | Single-task training, including the [CSI-Finetuning](csi-finetuning.md) experiments. |
| `src/main_sac_multi_task.py` | The [MT-SAC / MT-PPO baselines](mt-baselines.md). |

Below are example commands for the two primary training approaches.

## OBC from scratch

This starts an On-policy Behavioral Cloning (OBC) training from scratch using all 14 tasks.
It assumes expert demonstrations are available, since `imitation_coef > 0`.

```bash
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_cw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis \
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

## Final Arnold agent (self-distillation from super-experts)

This trains the final Arnold agent by imitating the super-expert policies produced by RL
fine-tuning, resuming from the pre-trained OBC model. Because `--use_expert_actions` is
absent, this is an OBC run — the rollouts come from the student.

```bash
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_cw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
    --load_path data/student_policies/obc \
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

## Configuring other training types

`src/main_bc_ppo_multi_task.py` covers the remaining training regimes through the loss
coefficients:

=== "PPO"

    Set `--imitation_coef=0`, `--pg_coef=1`, `--ent_coef=1e-6` and `--lr=2e-5`.

=== "OBC-PPO"

    Similar to PPO, but set `--imitation_coef=1` and `--lr=1e-3`.

=== "BC"

    Same setup as OBC (`--imitation_coef=1`, `--pg_coef=0`, `--ent_coef=0`, `--lr=1e-3`),
    but **add `--use_expert_actions`**. That flag steps the environments with the expert's
    actions instead of the student's, so the rollouts follow the expert's state
    distribution — which is exactly what makes it off-policy BC rather than OBC.

    Omitting the flag gives OBC. The *Final Arnold agent* command above has no
    `--use_expert_actions`, so it is an OBC run, not BC.

=== "Super-experts (RL fine-tuning)"

    Same as PPO, but resume from an OBC checkpoint with `--load_path`, and train one task at
    a time: `--num_envs_per_task=32 --tasks <single_task_name>`.

    Two settings matter here. Lower the learning rate to `--lr=2e-6`, and **reset the action
    distribution's standard deviation** with `--reset_std --log_std_init -6.9` — during
    pre-training the standard deviation decays towards 0, and without resetting it to
    ~`1e-3` the fine-tuning cannot explore enough to improve on the expert.

    `--log_std_init` is the *log* standard deviation, so `1e-3` corresponds to
    `ln(1e-3) ≈ -6.9`.

## Coefficient cheat sheet

Values are the paper's Table 5.

| Regime | `imitation_coef` | `pg_coef` | `ent_coef` | `vf_coef` | `lr` | Expert rollout |
| --- | --- | --- | --- | --- | --- | --- |
| BC | 1 | 0 | 0 | 0.5 | `1e-3` | Yes (`--use_expert_actions`) |
| OBC | 1 | 0 | 0 | 0.5 | `1e-3` | No |
| OBC-PPO | 1 | 1 | `1e-6` | 0.5 | `1e-3` | No |
| PPO | 0 | 1 | `1e-6` | 0.5 | `2e-5` | n/a |
| RL fine-tuning | 0 | 1 | `1e-6` | 0.5 | `2e-6` | n/a |

Shared across every regime: batch size 128, rollout steps 512, 3 epochs, gamma 0.99, GAE
lambda 0.95, clip range 0.2, max gradient norm 0.5, MSE imitation loss, advantage and
observation standardization on. Initial standard deviation is 1.0 everywhere except RL
fine-tuning, which resets it to `1e-3`.

## Training schedule

The paper's runs are two phases, which the commands above do not show as a single step:

1. **50M steps** at the initial learning rate.
2. **A further 5M steps** at a reduced learning rate of `1e-5` — for BC, OBC and OBC-PPO
   only. This is a separate invocation resuming with `--load_path` and `--lr=1e-5`. PPO and
   MT-PPO do not get this phase.

!!! note "`--num_steps` counts additional steps"
    When resuming with `--load_path`, the step counter is not reset, and `--num_steps` is
    the number of steps to run *on top of* the loaded checkpoint.
