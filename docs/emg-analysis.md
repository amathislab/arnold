# Human EMG and gait factors

Use the [Arnold environment](installation.md). Run the offline analyses:

```bash
python plotting/analyze_emg.py
python plotting/analyze_gait_factors.py
```

Inputs are `data/analysis/emg/human.npz` and `simulation.npz`. Use `--data_dir`
and `--out_dir` to change input and output paths. CSVs and SVGs are written to
`data/figures/emg/` and `data/figures/gait_factors/`.

The supplied `simulation.npz` can be analyzed directly. The recordings under
`data/analysis/emg/rollouts/` can also be processed with `segment_gait.py` to regenerate it.

To collect new deterministic gait recordings, obtain the models using [Data](data.md),
then supply the actual checkpoint paths:

```bash
python plotting/collect_gait.py --load path/to/arnold/rl_model_64670238_steps.zip --num_episodes 300 --moving_goal --output data/analysis/emg/rollouts/arnold.h5
python plotting/collect_gait.py --load path/to/obc/rl_model_54974700_steps.zip --num_episodes 150 --output data/analysis/emg/rollouts/obc.h5
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

## Importing human profiles

The importer first loads subjects 04–12 at 4.5 km/h directly from the Mathis Lab
[Kinesis assets dataset](https://huggingface.co/datasets/amathislab/kinesis-assets/tree/main/emg_assets/human_emg):

```bash
python src/analysis/import_human_emg.py
```

It validates the published `EMG_labels.npy` against `data/analysis/reproduction/emg_muscles.json`
and creates `data/analysis/emg/human.npz` with shape `(9, 100, 9)`:
subjects × gait-cycle samples × muscles. These inputs are already processed; raw Zenodo
preprocessing is unnecessary. Downloads are read in memory; this command writes the packed
NPZ, not a new local cache of subject files.

If downloading fails, the importer uses a complete local cohort from
`data/analysis/emg/human_profiles/` (or `--input_dir`). A cohort consists of
`EMG_subject_04_walk_45_avg.npy` through `EMG_subject_12_walk_45_avg.npy`, each with a time
column followed by the nine configured muscles. It never mixes downloaded and local
subjects. Invalid published data raises an error rather than triggering fallback.

To explicitly use local/custom profiles without a network request:

```bash
python src/analysis/import_human_emg.py --local-only --input_dir path/to/human_emg
```

Source URLs and SHA-256 checksums for the supplied local profiles are recorded in
`human_profiles/source.json`.

Simulation profiles are produced by `collect_gait.py` and `segment_gait.py`.

The cross-human reference uses each subject's correlation with the mean of the other
subjects, rather than the average of pairwise subject correlations. Policy statistics
pair subjects, Fisher-transform their
mean correlations, and apply Bonferroni correction across the three policy comparisons.
These EMG and factor analyses extend the supplied manuscript PDF rather than reproducing
an EMG figure contained in that version.
