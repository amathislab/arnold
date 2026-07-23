# Arnold

**A generalist muscle transformer policy.**

Arnold is a transformer policy trained to control musculoskeletal models across 14
manipulation and locomotion tasks spanning four embodiments. This site documents how to
install the code, download the released checkpoints, train new policies, evaluate the
released ones, and reproduce figures in the paper.

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    Docker image or conda environment. Under 30 minutes on a modern machine.

-   :material-database: **[Data and checkpoints](data.md)**

    What ships in the repository, and what you need to fetch from Zenodo.

-   :material-dumbbell: **[Training](training.md)**

    BC, PPO, OBC, OBC-PPO, RL fine-tuning and self-distillation.

-   :material-chart-line: **[Evaluation](evaluation.md)**

    Benchmark the released OBC, Arnold and expert policies.

</div>

## What is included

1. The code and scripts to train policies with **BC, PPO, OBC, OBC-PPO, RL fine-tuning and
   self-distillation** — including the expert policies used for imitation learning.
2. **Pretrained checkpoints** for every method in the paper, including the ablations, so the
   results and videos can be reproduced directly.

## Model checkpoints and benchmark results

All checkpoints, benchmark result files and cached learning curves are hosted on Zenodo:

[:material-database: Zenodo record 21493316](https://zenodo.org/records/21493316){ .md-button .md-button--primary }

The git repository contains only the code plus small configuration files. Everything else
must be unzipped into `data/` before running the training, evaluation or plotting scripts —
see [Data and checkpoints](data.md) for the exact directory layout.

## Reproducing the paper

| Result | Page |
| --- | --- |
| Radar plot, ablation bar plots, and all learning curves | [Replicate plots](replicate-plots.md) |
| Effective dimensionality of the learned actions | [CSI analysis](csi-analysis.md) |
| Training inside a constrained action subspace | [CSI-Finetuning](csi-finetuning.md) |
| MT-SAC vs. MT-PPO | [Multi-task RL baselines](mt-baselines.md) |

All figures are written under `data/figures/`.

## Citation

If you use Arnold in your research, please cite:

> Chiappa, A. S., An, B., Simos, M., Li, C., & Mathis, A. (2025).
> *Arnold: a generalist muscle transformer policy.* arXiv:2508.18066.
> [:material-file-document: arXiv](https://arxiv.org/abs/2508.18066) ·
> [:material-file-pdf-box: PDF](https://arxiv.org/pdf/2508.18066)

```bibtex
@article{chiappa2025arnold,
  title         = {Arnold: a generalist muscle transformer policy},
  author        = {Chiappa, Alberto Silvio and An, Boshi and Simos, Merkourios and
                   Li, Chengkun and Mathis, Alexander},
  journal       = {arXiv preprint arXiv:2508.18066},
  year          = {2025},
  eprint        = {2508.18066},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2508.18066}
}
```
