# Collect analysis signals

Download models with `python scripts/fetch_data.py --profile models`.

```bash
python plotting/collect_activations.py \
    --load data/final_checkpoints/arnold/seed_0/rl_model_64670238_steps.zip \
    --task baoding_p1_ccw --arnold --normalize --signals_only \
    --num_episodes 100 --seed 0 --device cpu \
    --policy_id arnold/seed_0 --out_dir data/analysis/signals
```

For a specialist expert:

```bash
python plotting/collect_activations.py --expert --task baoding_p1_ccw \
    --deterministic --signals_only --num_episodes 100 --seed 0 --device cpu \
    --policy_id expert --out_dir data/analysis/signals
```

Each `<policy_id>/<task>.h5` contains episode groups with policy actions, action
means, raw joint positions, rewards and solved flags. Joint and muscle names give
column order. Episode `i` uses seed `--seed + i`. Add `--deterministic` for mean
actions, and `--physical_signals` for actuator controls and muscle activation state.
Policy actions and joints precede the step; physical signals follow it. Each
dataset records its timing. Use `--num_success 100 --max_attempts 1000` to collect
100 episodes with at least one solved step. Analysis selections are in
`data/reproduction/signals.json`.

Export existing hand recordings:

```bash
python scripts/export_signals.py \
    --input_dir data/activations/arnold/seed_0 \
    --output data/analysis/signals/arnold/seed_0/baoding_p1_ccw.h5 \
    --task baoding_p1_ccw --policy_id arnold/seed_0 --horizon 200 \
    --num_episodes 100 --signals actions joint_positions
```

Use `--episodes <ids>` to select recorded episodes. Exported `muscle_controls`
come from the recordings' `activations` dataset and precede the step.
