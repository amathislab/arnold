# Reproduce tables and performance plots

Follow the [installation instructions](installation.md) and download benchmark results:

```bash
python scripts/fetch_data.py --profile benchmarks
python plotting/ablation_table.py
python plotting/plot_radar.py
python plotting/plot_arnold_ablation_bars.py
python plotting/plot_ppo_ablation_bars.py
python plotting/plot_relative_dotplot.py
python plotting/plot_capacity_performance.py
python plotting/plot_csi_analysis.py
python plotting/plot_bilateral_reward.py
python plotting/plot_mt_algos.py
```

Tables use solved-step fractions relative to expert performance, with SEM across seeds and task-paired Wilcoxon tests with Holm correction. The table command saves CSV and LaTeX files under `data/analysis/tables/`. Plots are saved under `data/figures/`; CSI also writes per-run values and run counts.

## Learning curves

```bash
python plotting/plot_rl_finetuning_curves.py
```

The included CSV contains the OBC training curves and four specialist PPO curves, plotted as solved fractions with a 101-point Savitzky–Golay filter (order 2). Plot another CSV with:

```bash
python plotting/plot_learning_curves.py --curves path/to/curves.csv --panel transfer
```

CSV columns are `panel,method,seed,task,stage,step,value,metric`. Use `metric=solved_fraction` or `metric=reward` for the recorded scalar. The general plot command scales solved fractions relative to experts; add `--raw` to plot their original values. It uses a five-point moving average; set `--window` to change it. Set stage offsets explicitly, for example `--offsets transfer=50000000`. Multiple seeds show mean ± standard deviation over shared recorded steps.

Export a scalar from TensorBoard logs, supplying event files in resume order:

```bash
python scripts/export_learning_curves.py \
  --events path/to/events.out.tfevents.* --tag MuscleDieReorientP0-v0/solved \
  --panel transfer --method Transfer --task reorient --stage transfer --seed 0 \
  --output data/analysis/curves/transfer_reorient.csv
```
