# Data and checkpoints

The git repository contains only the code plus the small configuration files. Everything
else lives on [Zenodo](https://zenodo.org/records/21493316) and must be unzipped into the
`data/` directory before running the training, evaluation or plotting scripts.

## Included in the repository

No download needed for these:

| Directory | Contents |
| --- | --- |
| `data/env_configs/` | Per-task environment configuration JSONs (`ENV_CONFIG_PATH`). |
| `data/expert_configs/` | Expert-policy configuration JSONs (`EXPERT_CONFIG_PATH`). |

## Downloaded from Zenodo

Find and download these from the Zenodo record at
[https://zenodo.org/records/21493316](https://zenodo.org/records/21493316), then unzip them
into `data/`:

| Directory | Contents | Needed for |
| --- | --- | --- |
| `data/student_policies/` | Trained OBC / Arnold / single- and multi-task student policy checkpoints (`.zip` + `vecnormalize.pkl`) and their TensorBoard training logs. | Evaluation (`src/benchmark.py`), activation collection (`plotting/collect_activations.py`), and the student / transfer learning-curve plots. |
| `data/expert_policies/` | (Super-)expert policy checkpoints and their TensorBoard logs (`EXPERT_POLICIES_PATH`). | Expert evaluation (`src/benchmark.py --expert`) and `plotting/plot_rl_finetuning_curves.py`. |
| `data/final_benchmarks/` | Per-method, per-seed benchmark result JSONs, plus `expert_policies/` result JSONs used as the baseline. | The paper's radar and ablation bar plots, and the expert baseline in every performance plot. |
| `data/final_benchmarks_extra/` | CSI (`csi_*`), bilateral, and the PPO w/o-rew-norm benchmark result JSONs, plus the cached MT-SAC / MT-PPO learning curves in `mt-curves/`. | `plotting/plot_csi_analysis.py`, `plotting/plot_csi_curves.py`, the PPO ablation plot, and `plotting/plot_mt_algos.py`. |
| `data/kinesis/` | MuJoCo model assets for the `kinesis` locomotion task. | Any run that instantiates the `kinesis` environment. |

## Resulting layout

```text
data/
├── env_configs/              # in repo
├── expert_configs/           # in repo
├── student_policies/         # Zenodo
├── expert_policies/          # Zenodo
├── final_benchmarks/         # Zenodo
├── final_benchmarks_extra/   # Zenodo
│   └── mt-curves/
└── kinesis/                  # Zenodo
```

Scripts also write into `data/` as they run — `data/activations/`, `data/pca_analysis/` and
`data/figures/` are created on demand.

## Weights & Biases

!!! note "`plot_mt_algos.py` runs offline"
    `plotting/plot_mt_algos.py` reads its MT-SAC / MT-PPO learning curves from the cached
    CSVs in `data/final_benchmarks_extra/mt-curves/` (included in the `final_benchmarks_extra`
    download), so it runs without wandb access.

    Any missing curve is re-fetched from Weights & Biases automatically and re-cached, which
    requires a logged-in `wandb` account with access to the runs referenced in the script.
    External users may not have access to the original wandb runs, and they may eventually
    be deleted by the Arnold team.
