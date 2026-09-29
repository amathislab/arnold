# Data and checkpoints

The git repository contains code and environment/expert configuration files. Benchmark
results, model weights, training logs and analysis data are external inputs. The release
record is [Zenodo v3](https://zenodo.org/records/21807280).

Environment and expert configurations are already included under `data/env_configs/` and
`data/expert_configs/` without need to download from Zenodo. External inputs from Zenodo use this layout:

## External inputs

| Directory | Contents | Needed for |
| --- | --- | --- |
| `data/final_benchmarks/` | Per-method/per-seed result JSONs and expert references; `arnold_single_task/` and `transfer_learning/` logs; `example_training_curve/` base OBC logs; `example_checkpoint/` with one checkpoint and matching normalization file. | Performance, ablation, student, transfer and fine-tuning plots; checkpoint-loading examples. |
| `data/final_benchmarks_extra/` | CSI evaluation results and `csi*/training/` logs; bilateral and normalization ablations; `mt-curves/` caches; `rl_finetuning/` logs. | CSI, baseline and RL fine-tuning plots. |
| `data/analysis/` | Curve caches, `reproduction/` metadata, compact hand signals, human/simulation EMG inputs, gait recordings and exported tables. | Learning curves, PCA/NMF summaries, hand and EMG analyses. |
| `data/final_checkpoints/` | Separately released model checkpoints and associated normalization/configuration files. | Evaluation, fresh recordings and resumed training beyond the bundled example. |
| `data/expert_policies/` | Expert policy checkpoints (`EXPERT_POLICIES_PATH`). | Expert evaluation and rollout collection. |
| `data/kinesis/` | Locomotion model assets. | Instantiating the Kinesis environment. |

Extract the packages from the repository root:

```bash
mkdir -p data
tar -xzf final-benchmarks.tar.gz -C data
tar -xzf final-benchmarks-extra.tar.gz -C data
tar -xzf expert-policies.tar.gz -C data
tar -xzf analysis.tar.gz -C data
```

The single-task and transfer folders contain training logs and configurations, not model
checkpoints. The bundled example checkpoint is
`data/final_benchmarks/example_checkpoint/rl_model_64670238_steps.zip`, accompanied by
`rl_model_vecnormalize_64670238_steps.pkl`. Other checkpoint paths require the separate
model release. Training logs alone cannot generate new rollout recordings.

## Resulting layout

```text
data/
├── env_configs/                    # in Git
├── expert_configs/                 # in Git
├── final_benchmarks/
│   ├── arnold_single_task/          # per-task training logs
│   ├── transfer_learning/           # eight transfer/scratch runs
│   ├── example_training_curve/      # base OBC logs
│   ├── example_checkpoint/          # one model + normalization/configs
│   ├── expert_policies/             # reference JSONs, not expert weights
│   └── <method>/seed_<n>/           # benchmark results
├── final_benchmarks_extra/
│   ├── csi*/training/               # CSI logs beside evaluation data
│   ├── rl_finetuning/               # Arnold fine-tuning logs
│   └── mt-curves/                   # cached MLP baseline curves
├── analysis/
│   ├── reproduction/               # selections and references
│   ├── signals/                    # supplied compact hand recordings
│   ├── emg/                        # human/simulation profiles and gait rollouts
│   ├── historical_curves.csv
│   └── learning_curves.csv.gz
├── final_checkpoints/              # separate download
├── expert_policies/                # separate download: weights
└── kinesis/                        # separate download: model assets
```

Recordings, fresh intervention results and figures are generated under `data/activations/`,
`data/pca_analysis/` and `data/figures/`, respectively.

## Weights & Biases

!!! note "`plot_mt_algos.py` runs offline"
    `plotting/plot_mt_algos.py` reads its MT-SAC / MT-PPO learning curves from the cached
    CSVs in `data/final_benchmarks_extra/mt-curves/` (included in the `final_benchmarks_extra`
    download), so it runs without wandb access.

    Any missing curve is re-fetched from Weights & Biases automatically and re-cached, which
    requires a logged-in `wandb` account with access to the runs referenced in the script.
    External users may not have access to the original wandb runs, and they may eventually
    be deleted by the Arnold team.

## Analysis inputs

`data/analysis/reproduction/` contains selected benchmark filenames, analysis cohorts,
EMG muscle mappings and human reference values. Shared task labels live in
`src/analysis/metadata.py`. Recording axes come from file metadata or the task model. Expert summaries under
`data/final_benchmarks/expert_policies/` provide the reference for relative performance.
Offline hand analyses default to supplied compact recordings in `data/analysis/signals/`.
`collect_activations.py` writes to `data/activations/`; intervention results use
`data/pca_analysis/`. `data/analysis/` holds EMG inputs, exported tables and portable
learning curves. The packaged `data/analysis/learning_curves.csv.gz` and
`data/analysis/historical_curves.csv`
files contain cached plot data, not raw recordings.

Collect/export hand recordings with [Analysis signals](signals.md). Human EMG requires
external preprocessed subject profiles; [EMG analysis](emg-analysis.md) documents their
format and import command.
