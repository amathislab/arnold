# Task-specific sensorimotor vocabulary ablation

The shared Arnold vocabulary assigns one embedding to each token across tasks.
The Task-SV ablation keeps the same token IDs and transformer but gives each
task an independent embedding table. Both sensory tokens and actuator tokens
use the table for their task. The paper's Task-SV comparison uses **learnable**
tables; frozen random tables are available as an additional diagnostic and
should be reported separately.

## Train matched models

Use the same task order, task sampling weights, hyperparameters, and random
seeds for both arms. Repeated task names below intentionally increase their
sampling weight; repetitions refer to the same embedding table.

```bash
TASKS=(
  hand_thumb_reach hand_index_reach hand_middle_reach hand_ring_reach
  hand_little_reach reorient pen baoding_p1_cw baoding_p1_ccw baoding_p2
  baoding_p2_overlap elbow_pose relocate kinesis kinesis relocate
  baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis relocate
  baoding_p1_ccw baoding_p2 baoding_p2_overlap kinesis kinesis
)
COMMON=(
  --tasks "${TASKS[@]}" --num_envs_per_task 2
  --ent_coef 0 --vf_coef 0.5 --pg_coef 0 --imitation_coef 1
  --num_steps 50000000 --batch_size 128 --rollout_steps 512
  --embedding_size 128 --dim_feedforward 512 --num_heads 4 --num_layers 6
  --lr 1e-3 --log_interval 1 --n_epochs 3
  --separate_vf_decoder --policy_outputs_variance --norm_reward
  --dense_reward --local
)

for seed in 0 1 2; do
  python src/main_bc_ppo_multi_task.py "${COMMON[@]}" \
    --positional_encoding learned --out_prefix obc_shared_ --seed "$seed"
  python src/main_bc_ppo_multi_task.py "${COMMON[@]}" \
    --positional_encoding task_specific --task_specific_learnable \
    --out_prefix obc_task_sv_ --seed "$seed"
done
```

`--task_specific_non_learnable` selects the separate frozen-table diagnostic.
The code saves `args.json`, `vocabulary.json`, the normalization state, and
checkpoints together in each run directory. Compare checkpoints at the same
training step. The archived runs have a `rl_model_49977000_steps.zip`
checkpoint for the shared seed 1 model and the learnable Task-SV seeds 0, 1,
and 2; the archive has no shared seeds 0 and 2 at this step.

## Evaluate

Evaluate each checkpoint with the same 14 tasks and 200 episodes per task.
`benchmark.py` reads the task list and observation history from the run's
`args.json`; it restores observation normalization with `--normalize` and
passes the original training task index to Task-SV checkpoints. The vocabulary type (task-specific or shared) is also loaded from the checkpoint. Repeated task
names are evaluated once. Use stochastic actions for comparison with the
paper; add `--deterministic` only for a separately labelled analysis.

```bash
RUN_NAME="<run_name>"
CKPT="output/training/ongoing/${RUN_NAME}/rl_model_49977000_steps.zip"
python src/benchmark.py \
  --load "$CKPT" --arnold --normalize --num_episodes 200 --seed 0 \
  --device cuda --save_results \
  --out_dir data/benchmarks/task_specific_ablations
```

Run the same command for each shared and Task-SV seed. Results are written as
`*_results.json` under the output directory. For each task and seed, compare
`avg_solved_step_frac`; to match the paper's relative performance, divide it
by the corresponding specialist expert's `avg_solved_step_frac`. Report the
mean and variation across seeds and keep the frozen-table diagnostic separate.

The archive's checkpoint handoff also describes an older seed 1 comparison
against **frozen** task-specific embeddings. It is a different ablation arm
from the learnable Task-SV model.
The training code in that archive selected the first task ID for a whole
mixed-task batch. This branch routes each sample through its own task table;
existing archive checkpoints should therefore be treated as exploratory
until their training behavior has been verified.
