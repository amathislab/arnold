# Arnold

**A generalist muscle transformer policy.**

Arnold is a transformer policy trained to control musculoskeletal models across 14
manipulation and locomotion tasks spanning four embodiments. This site documents how to
install the code, download the released checkpoints, train new policies, evaluate the
released ones, and reproduce figures in the paper.

<div class="grid cards" markdown>

-   :material-download: **[Installation](installation.md)**

    Set up Docker or a conda environment.

-   :material-database: **[Data and checkpoints](data.md)**

    Download checkpoints and benchmark results.

-   :material-dumbbell: **[Training](training.md)**

    BC, PPO, OBC, OBC-PPO, RL fine-tuning and self-distillation.

-   :material-chart-line: **[Evaluation](evaluation.md)**

    Benchmark the released OBC, Arnold and expert policies.

</div>

## Download data

Download the [Zenodo v3 release](https://zenodo.org/records/21807280) from the repository root:

```bash
python scripts/fetch_data.py --profile all
```

See [Data and checkpoints](data.md) for individual download profiles and paths.

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
