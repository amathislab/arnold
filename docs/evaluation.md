# Evaluation

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
