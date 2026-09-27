# Hand smoothness and Baoding PCA

Use the [Arnold environment](installation.md) and place the compact recordings
under `data/analysis/signals/`. To extract the signal archive:

```bash
tar -xzf arnold-analysis-signals.tar.gz -C data
```

Run both analyses from the repository root:

```bash
python plotting/analyze_smoothness.py
python plotting/analyze_baoding_kinematics.py
```

Use `--data_dir <signals_directory>` and `--out_dir <results_directory>` to change
the paths. See [Analysis signals](signals.md) to collect new recordings.

Smoothness writes per-episode metrics, per-task means, Wilcoxon/Holm comparisons
and `smoothness.svg` to `data/figures/smoothness/`. Derivatives use one simulation
step; comparisons pair the 11 task means, with Holm correction within each metric.

Baoding PCA writes dimensionality counts, cumulative variance curves, the human
comparison and `baoding_p1_ccw.svg` to `data/figures/baoding_pca/`. PCA centers raw
joint coordinates. Velocities are finite differences within each episode.
Dimensionality is `(PCs at 85% + PCs at 95%) / 2`, separately for position and velocity.
The human values in `data/reproduction/baoding_human.csv` use the Ball rows,
Angle scaling, 20 joints in Tables 1–2 of
[Todorov and Ghahramani (2004)](https://roboti.us/lab/papers/TodorovEMBC04.pdf).
