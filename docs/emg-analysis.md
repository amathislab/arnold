# Human EMG and gait factors

Use the [Arnold environment](installation.md). Run the offline analyses:

```bash
python plotting/analyze_emg.py
python plotting/analyze_gait_factors.py
```

Inputs are `data/analysis/emg/human.npz` and `simulation.npz`. Use `--data_dir`
and `--out_dir` to change input and output paths. CSVs and SVGs are written to
`data/figures/emg/` and `data/figures/gait_factors/`.

To collect new deterministic gait recordings, download the models:

```bash
python scripts/fetch_data.py --profile models
python plotting/collect_gait.py --load data/final_checkpoints/arnold/seed_0/rl_model_64670238_steps.zip --num_episodes 300 --moving_goal --output data/analysis/emg/rollouts/arnold.h5
python plotting/collect_gait.py --load data/final_checkpoints/obc/seed_0/rl_model_54974700_steps.zip --num_episodes 150 --output data/analysis/emg/rollouts/obc.h5
python plotting/collect_gait.py --expert --num_episodes 150 --output data/analysis/emg/rollouts/kinesis.h5
python plotting/segment_gait.py
```

Collection uses seed 42 and a goal 2 m ahead: moving for Arnold, fixed at reset
for OBC and Kinesis. Segmentation uses
30 Hz recordings, heel-contact threshold 300, a 5 Hz contact/velocity filter,
the central 80% of cycle durations and mean forward speed ≥0.2 m/s. Muscle state
is resampled directly to 200 points. EMG uses left-heel cycles and biceps femoris
long head; factors use right-heel cycles and sum gluteal compartments.

Human profiles use subjects 04–12 walking at 4.5 km/h from
[Wang et al., Comprehensive Kinetic and EMG Dataset of Daily Locomotion](https://zenodo.org/records/7422031),
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The profiles are derived gait-cycle averages of the nine recorded muscles.
