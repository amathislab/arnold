# Control subspace analysis

Install Arnold using the [installation instructions](installation.md), download the [analysis signals](signals.md), and install `scikit-learn` for fitting subspaces.

```bash
python plotting/plot_action_pca_variance.py
python plotting/analyze_subspaces.py
python plotting/plot_pca_inactivation.py
```

PVD is directed from the first condition to the second. PAD is the mean principal angle in degrees. Subspace plots show mean ± standard deviation over task pairs or policy pairs.

To collect 100 successful episodes per task for the capacity comparison, use the [signal collector](signals.md) with `--num_success 100 --max_attempts 1000`, then run:

```bash
python plotting/analyze_subspaces.py --selection capacity_recordings --successful 100
```

Run fresh interventions with a downloaded Arnold checkpoint:

```bash
python plotting/analyze_pca_inactivation.py \
  --load data/final_checkpoints/arnold/seed_0/rl_model_64670238_steps.zip \
  --signals data/analysis/signals/arnold_pca/seed_0 \
  --method pca --scope global --num_episodes 100 \
  --out_dir data/analysis/interventions/pca
python plotting/analyze_pca_inactivation.py \
  --load data/final_checkpoints/arnold/seed_0/rl_model_64670238_steps.zip \
  --signals data/analysis/signals/arnold_nmf/seed_0 \
  --method nmf --scope global --num_episodes 100 --n_fits 10 \
  --out_dir data/analysis/interventions/nmf
python plotting/plot_pca_inactivation.py --curves data/analysis/interventions/pca/curves.csv
```

Use `--scope task` for task-specific fitting. For expert-derived PCA, use `--signals data/analysis/signals/expert_pca --basis_policy expert`. PCA projects sampled actions around the fitted mean. NMF projects physical actuator controls after actuator processing, using `nndsvdar`, fit seeds 0–9, and 2,000 maximum iterations. Evaluation uses stochastic actions, frozen observation normalization, and episode seeds starting at 0.

Performance is the solved-step count divided by the fixed task horizon, then divided by expert performance. The saved curves contain 20 episodes per rank for task-specific PCA, 100 for global PCA, and 100 across ten fits for NMF. PCA spans ranks 1–39; NMF spans 1–38. Use `--dimensions` to select ranks for a fresh run. Commands save numerical CSVs and SVG figures.
