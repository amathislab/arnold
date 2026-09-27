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

## Final Arnold agent (self-distillation from super-experts)

Resume OBC from the 55M-step checkpoint using the super-expert configuration:

```bash
python src/main_bc_ppo_multi_task.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis kinesis \
        relocate baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis \
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

## Training variants

Use the OBC command above with these options:

| Variant | Options |
| --- | --- |
| PPO | `--imitation_coef 0 --pg_coef 1 --ent_coef 1e-6 --lr 2e-5` |
| OBC-PPO | `--imitation_coef 1 --pg_coef 1 --ent_coef 1e-6` |
| BC | `--use_expert_actions` |
| Task-SV OBC | `--positional_encoding task_specific` |
| OBC without observation normalization | `--ablate_obs_norm` |

For BC, OBC and OBC-PPO, continue for 5M steps with
`--load_path <run_directory> --num_steps 5000000 --lr 1e-5`.

For single-task PPO fine-tuning, resume from the 55M-step OBC checkpoint and use
`--tasks <task> --num_envs_per_task 32 --imitation_coef 0 --pg_coef 1
--ent_coef 1e-6 --lr 2e-6 --reset_std --log_std_init -3`.

## Atomic vocabulary PPO

Build a vocabulary for the training tasks:

```bash
python src/build_atomic_vocabulary.py \
    --tasks hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach hand_little_reach \
        reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2 baoding_p2_overlap elbow_pose relocate kinesis \
    --output data/atomic_vocabulary.json
```

Add `--vocabulary_mode atomic --atomic_vocabulary_path data/atomic_vocabulary.json`
to the PPO training command. Checkpoints save the mapping as `vocabulary.json`.
