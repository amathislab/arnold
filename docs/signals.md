# Analysis signals

The existing `plotting/collect_activations.py` continues to produce intermediate transformer
activations and per-episode HDF5 files under `data/activations/`. New analyses can read
those files directly when their configured policy directory matches the recording folder.
The same directory can also hold compact files under `data/activations/<policy_id>/<task>.h5`.

## Collect compact hand recordings

Use checkpoints from the current [data layout](data.md), including their neighboring
`args.json`, `vocabulary.json` and normalization checkpoint. Supply the actual model path:

```bash
python plotting/collect_signals.py \
    --load path/to/rl_model_64670238_steps.zip \
    --task baoding_p1_ccw --arnold --normalize \
    --num_episodes 100 --seed 0 --device cpu \
    --policy_id arnold/seed_0
python plotting/collect_signals.py --expert --task baoding_p1_ccw \
    --deterministic --num_episodes 100 --seed 0 --device cpu --policy_id expert
```

Repeat for every policy/task in `data/analysis/reproduction/signals.json`. That file specifies
analysis cohorts, not checkpoint download paths. Use each analysis's `--selections` option
to provide a JSON with your local policy IDs. For legacy recordings use
`--data_dir data/activations` and map policy IDs to the existing recording subdirectories.

Each compact file has `episode_N` groups with `actions`, `action_means`, `joint_positions`,
`rewards`, and `solved`. Joint/muscle names specify column order; metadata records seed,
fixed horizon and timestep. Actions and positions precede the environment step. Add
`--physical_signals` for `muscle_controls` and `muscle_activations` measured after the step.
These are distinct from policy actions. Add `--num_success 100 --max_attempts 1000` to
retain successful episodes only. Failed attempts retain their episode IDs in the seed sequence.

This command supports the 11 MyoHand tasks with the current compositional policy.
For locomotion use [collect_gait.py](emg-analysis.md). It does not require atomic vocabulary
or Task-SV changes from the rolled-back training branch.

## Export existing recordings

```bash
python src/analysis/export_signals.py \
    --input_dir data/activations/285_64670238 \
    --output data/activations/arnold/seed_0/baoding_p1_ccw.h5 \
    --task baoding_p1_ccw --policy_id arnold/seed_0 --horizon 200 \
    --num_episodes 100 --signals actions action_means joint_positions
```

Set `--horizon` and `--dt` to the recording's environment settings. `--episodes` selects
specific episode IDs. The importer reads raw joint positions from the legacy observation
layout and flattens singleton action dimensions. `muscle_controls` can be exported only
when the original file actually contains them (or the legacy physical `activations`
dataset); they cannot be reconstructed from policy actions alone. Legacy physical controls
are marked `before_step`, following their original recording convention.

No signal archive is assumed to be available. Collect recordings or import existing files
before running the hand/subspace analyses.

Export obtains joint and muscle column names from recording metadata, or from the
current task model when older recordings omit them. For recordings from a different
model, supply `--axes path/to/axes.json` containing `joint_names` and `muscle_names`.
Analysis can read existing per-episode recordings directly; conversion is optional.
