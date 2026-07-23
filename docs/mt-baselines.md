# Multi-task RL baselines (MT-SAC vs. MT-PPO)

Both baselines are trained with the same script, `src/main_sac_multi_task.py` (multi-task
SAC/PPO with an MLP policy); MT-PPO is selected with `--algo ppo`.

## 1. Training

=== "MT-SAC"

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

=== "MT-PPO"

    The same script with `--algo ppo` and a rollout length:

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

## 2. Benchmarking

Both baselines are evaluated with `src/benchmark_multi_task_mlp.py`. The task order,
environment id and algorithm are recovered from the `args.json` saved next to each
checkpoint, so only the checkpoint path is needed.

Evaluate the latest checkpoint of each seed:

```bash
python src/benchmark_multi_task_mlp.py \
    --load <run_dir>/rl_model_<steps>_steps.zip \
    --num_episodes 200 \
    --deterministic \
    --device cpu \
    --save_results \
    --out_dir data/final_benchmarks/mt_ppo/seed_0
```

## Related figures

| Figure | Script | Output |
| --- | --- | --- |
| MT-SAC vs. MT-PPO learning curves | [`plotting/plot_mt_algos.py`](replicate-plots.md#multi-task-rl-baselines-mt-sac-vs-mt-ppo) | `data/figures/mt_algos_training_curves.png` / `.svg` |
| MT-PPO bar of the PPO ablation plot | [`plotting/plot_ppo_ablation_bars.py`](replicate-plots.md#ppo-ablation-bar-plot) | `data/figures/bar_plot_ppo_ablation.png` / `.svg` |

Both scripts are documented on [Replicate plots](replicate-plots.md). The PPO ablation plot
picks up this baseline by aggregating `data/final_benchmarks/mt_ppo/`, so run the
benchmarking step above before regenerating it.
