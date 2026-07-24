import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter
import wandb

mpl.rcParams.update({"font.size": 20, "axes.titlesize": 20, "axes.labelsize": 20, "xtick.labelsize": 20, "ytick.labelsize": 20})

sys.path.insert(0, os.path.dirname(__file__))
from definitions import ROOT_DIR

# Local cache of the wandb learning curves (one CSV per run) so this script can
# run without live wandb access. Missing runs are fetched from wandb on demand
# and cached here (delete a CSV to force a refresh).
MT_CURVES_DIR = os.path.join(ROOT_DIR, "data", "final_benchmarks_extra", "mt-curves")


def run_path_to_filename(path):
    """Turn a wandb run path (/entity/project/run_id) into a flat CSV filename."""
    return path.strip("/").replace("/", "__") + ".csv"

# Maps wandb env ID to the task name used in expert benchmark files
ENV_ID_TO_TASK = {
    "MuscleBaodingP1-v1":             "baoding_p1_cw",
    "MuscleBaodingP2-v1":             "baoding_p2",
    "MuscleBaodingP2Overlap-v1":      "baoding_p2_overlap",
    "MuscleDieReorientP0-v0":         "reorient",
    "MuscleElbowPoseRandom-v0":       "elbow_pose",
    "MuscleHandIndexReachRandom-v0":  "hand_index_reach",
    "MuscleHandLittleReachRandom-v0": "hand_little_reach",
    "MuscleHandMiddleReachRandom-v0": "hand_middle_reach",
    "MuscleHandRingReachRandom-v0":   "hand_ring_reach",
    "MuscleHandThumbReachRandom-v0":  "hand_thumb_reach",
    "MuscleKinesis-v0":               "kinesis",
    "MusclePenTwirl-v0":              "pen",
    "MuscleRelocate-v0":              "relocate",
}

# Maps the wandb-logged env ID (prefix before -v0/-v1) to a display name
ENV_ID_TO_DISPLAY = {
    "MuscleBaodingP1-v1":           "Baoding CW/CCW",
    "MuscleBaodingP2-v1":           "Baoding harder",
    "MuscleBaodingP2Overlap-v1":    "Baoding hard",
    "MuscleDieReorientP0-v0":       "Die reorient",
    "MuscleElbowPoseRandom-v0":     "Elbow pose",
    "MuscleHandIndexReachRandom-v0":  "Index reach",
    "MuscleHandLittleReachRandom-v0": "Little reach",
    "MuscleHandMiddleReachRandom-v0": "Middle reach",
    "MuscleHandRingReachRandom-v0":   "Ring reach",
    "MuscleHandThumbReachRandom-v0":  "Thumb reach",
    "MuscleKinesis-v0":             "Walk to point",
    "MusclePenTwirl-v0":            "Pen reorient",
    "MuscleRelocate-v0":            "Object relocation",
}

# Each algorithm has several seeds; because of cluster preemption a single seed
# may be split across multiple wandb runs. Express this as:
#   "seeds": [ <seed>, <seed>, ... ]   where each <seed> is a LIST of run paths.
# The runs in a seed list are stitched (in order) into one continuous curve, and
# the seeds are then aggregated into a mean +- std band.
RUN_CONFIGS = {
    "MT-SAC": {
        "color": "#4C72B0",
        "seeds": [
            ["/boshi-an/arnold-new-exp/xsahxytq",
             "/boshi-an/arnold-new-exp/m1tbvp4y",
             "/boshi-an/arnold-new-exp/qcecj6ud"],
            ["/boshi-an/arnold-new-exp/xxd7xe3y"],
            ["/boshi-an/arnold-new-exp/0uvehx5f",
             "/boshi-an/arnold-new-exp/qrs8ns9l",
             "/boshi-an/arnold-new-exp/ji1px09d",
             "/boshi-an/arnold-new-exp/29k1vyxg",
             "/boshi-an/arnold-new-exp/c2d4ki7e",
             "/boshi-an/arnold-new-exp/gijw78rf"],
        ],
    },
    "MT-PPO": {
        "color": "#C44E52",
        "seeds": [
            ["/boshi-an/arnold-mtppo-new/x8d4gvsm",
             "/boshi-an/arnold-mtppo-new/9jj85mce",
             "/boshi-an/arnold-mtppo-new/7bf6ggaj",
             "/boshi-an/arnold-mtppo-new/pjaon8i8"],
            ["/boshi-an/arnold-mtppo-new/ktcc54v7"],
            ["/boshi-an/arnold-mtppo-new/7za2nszx",
             "/boshi-an/arnold-mtppo-new/ub4o4jxp"],
        ],
    },
}

# wandb column used for the x-axis (already in environment steps).
X_KEY = "global_step"

# Floor for the y-axis top, so tasks that stay near zero still get readable percent
# ticks instead of a column of 0%.
Y_MAX_MIN = 10.0
X_MAX = 5e7

SMOOTH_WINDOW = 20

# Stitch successive runs of a seed by offsetting each run to continue right after
# the previous one. Set False if the runs already share a continuous global step
# axis (then they are simply concatenated and sorted).
OFFSET_RUNS = True

# Number of points on the common grid used to aggregate seeds into mean +- std.
N_GRID = 300

# Also draw each individual seed curve faintly behind the mean band.
SHOW_INDIVIDUAL_SEEDS = True


def smooth(values, window=SMOOTH_WINDOW):
    if len(values) < window:
        return values, np.arange(len(values))
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    # align to the center of each window
    offset = window // 2
    indices = np.arange(offset, offset + len(smoothed))
    return smoothed, indices


def fetch_history(run, keys):
    # Fetch the task metrics. Requesting X_KEY together with them would return an
    # empty intersection, because global_steps is logged on different rows than
    # the per-task solved metrics.
    df = run.history(keys=list(keys), samples=5000, pandas=True)
    if df.empty or "_step" not in df.columns:
        return df

    # Fetch global_steps separately and map each metric row to it by interpolating
    # along the shared wandb _step axis (i.e. the closest logged global step).
    gdf = run.history(keys=[X_KEY], samples=5000, pandas=True)
    if X_KEY in gdf.columns:
        g = gdf[["_step", X_KEY]].dropna().sort_values("_step")
        if not g.empty:
            df[X_KEY] = np.interp(
                df["_step"].values.astype(float),
                g["_step"].values.astype(float),
                g[X_KEY].values.astype(float),
            )
            return df
    print(f"  WARNING: '{X_KEY}' not available for run '{run.name}', it will be skipped.")
    return df


def load_run_history(path):
    """Load one run's history from the local cache in ``MT_CURVES_DIR``.

    Falls back to a live wandb fetch (and caches the result) when the CSV is not
    present, so the plot works offline once the curves have been cached."""
    csv_path = os.path.join(MT_CURVES_DIR, run_path_to_filename(path))
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    print(f"  [{path}] no local cache at {csv_path}, fetching from wandb...")
    run = wandb.Api().run(path)
    solved_keys = [k for k in run.summary.keys() if "solved" in k.lower()]
    task_keys = [
        k for k in sorted(solved_keys)
        if "/" in k and k.rsplit("/", 1)[0] in ENV_ID_TO_DISPLAY
    ]
    df = fetch_history(run, task_keys)
    os.makedirs(MT_CURVES_DIR, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


def load_expert_scores():
    """Return, per env ID, the expert reference needed to normalise a logged curve.

    The wandb "<env_id>/solved" scalar is the fraction of solved steps per episode,
    so it is rescaled into solved steps by the episode length before being compared
    against the expert's avg_solved_steps (the same ratio plot_radar.py computes).
    Dividing by avg_solved_step_frac does exactly that in one step.
    """
    expert_dir = os.path.join(ROOT_DIR, "data", "final_benchmarks", "expert_policies")
    expert_results = {}
    for fname in os.listdir(expert_dir):
        if fname.endswith("_results.json"):
            with open(os.path.join(expert_dir, fname)) as f:
                expert_results.update(json.load(f))
    scores = {}
    for env_id, task in ENV_ID_TO_TASK.items():
        if task in expert_results:
            scores[env_id] = expert_results[task].get("avg_solved_step_frac", None)
    return scores


def get_seed_lists(config):
    """Return the list of seeds (each a list of run paths) for a method config,
    tolerating the older single-run 'path' format."""
    if "seeds" in config:
        seeds = config["seeds"]
    elif "runs" in config:  # flat list -> one run per seed
        seeds = [[r] for r in config["runs"]]
    elif "path" in config:  # single run -> single seed
        seeds = [[config["path"]]]
    else:
        raise ValueError(f"Run config {config} has no 'seeds', 'runs' or 'path'.")
    # Drop empty seeds (e.g. all-commented placeholders)
    return [s for s in seeds if s]


def run_series(df, key):
    """Extract a sorted (env_steps, values) series for one metric from a run df,
    using X_KEY (global_steps) as the x-axis."""
    if df is None or key not in df.columns or X_KEY not in df.columns:
        return None
    series = df[[X_KEY, key]].dropna()
    if series.empty:
        return None
    steps = series[X_KEY].values.astype(float)
    values = series[key].values.astype(float)
    order = np.argsort(steps, kind="stable")
    return steps[order], values[order]


def concat_seed(run_dfs, key, offset_runs=OFFSET_RUNS):
    """Stitch the runs of a single seed into one continuous (steps, values) curve."""
    parts_steps, parts_values = [], []
    last_end = 0.0
    for df in run_dfs:
        res = run_series(df, key)
        if res is None:
            continue
        steps, values = res
        if offset_runs:
            # Make this run continue right after the previous one.
            steps = steps - steps.min() + last_end
        parts_steps.append(steps)
        parts_values.append(values)
        last_end = steps.max()
    if not parts_steps:
        return None
    steps = np.concatenate(parts_steps)
    values = np.concatenate(parts_values)
    order = np.argsort(steps, kind="stable")
    return steps[order], values[order]


def aggregate_seeds(seed_curves, x_max=X_MAX, n_points=N_GRID):
    """Interpolate seed curves onto a common grid and return (grid, mean, std).

    Seeds that do not cover a given grid point are ignored at that point (so the
    band is defined wherever at least one seed has data)."""
    valid = [(s, v) for s, v in seed_curves if s is not None and len(s) > 1]
    if not valid:
        return None
    # Span the grid over the union of seed ranges so there are no leading/trailing
    # all-NaN columns (which would otherwise trigger nanmean "empty slice" warnings).
    lo = min(s.min() for s, _ in valid)
    hi = max(s.max() for s, _ in valid)
    if x_max is not None:
        hi = min(hi, x_max)
    if hi <= lo:
        return None
    grid = np.linspace(lo, hi, n_points)
    mat = np.full((len(valid), n_points), np.nan)
    for i, (steps, values) in enumerate(valid):
        covered = (grid >= steps.min()) & (grid <= steps.max())
        mat[i, covered] = np.interp(grid[covered], steps, values)
    valid_cols = np.any(~np.isnan(mat), axis=0)  # guard against gaps between seeds
    mean = np.full(n_points, np.nan)
    std = np.full(n_points, np.nan)
    with np.errstate(invalid="ignore"):
        mean[valid_cols] = np.nanmean(mat[:, valid_cols], axis=0)
        std[valid_cols] = np.nanstd(mat[:, valid_cols], axis=0)
    keep = ~np.isnan(mean)
    return grid[keep], mean[keep], std[keep]


def create_plot():
    expert_scores = load_expert_scores()

    # Map every method -> list of seeds, each seed a list of run paths.
    method_seeds = {name: get_seed_lists(cfg) for name, cfg in RUN_CONFIGS.items()}

    # Load each unique run's history from the local cache in
    # data/final_benchmarks_extra/mt-curves (falling back to wandb if missing).
    unique_paths = sorted({p for seeds in method_seeds.values()
                           for seed in seeds for p in seed})
    print(f"Loading {len(unique_paths)} unique run(s) from {MT_CURVES_DIR}...")
    run_hist = {}
    for path in unique_paths:
        df = load_run_history(path)
        run_hist[path] = df
        print(f"  [{path}] DataFrame shape: {df.shape}")

    # Use keys that appear in at least one run and match known env IDs.
    all_keys = set()
    for df in run_hist.values():
        all_keys.update(df.columns)
    task_keys = [k for k in sorted(all_keys)
                 if "/" in k and k.rsplit("/", 1)[0] in ENV_ID_TO_DISPLAY]
    print(f"\nUsing {len(task_keys)} task keys for plotting: {task_keys}")

    def key_to_display(key):
        return ENV_ID_TO_DISPLAY.get(key.rsplit("/", 1)[0], key)

    def relative(values, env_id):
        expert = expert_scores.get(env_id)
        if expert and expert > 0:
            return (values / expert) * 100
        print(f"Warning: no expert reference for '{env_id}', curve left unnormalised.")
        return values

    ncols = 4
    n_plots = len(task_keys)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    axes_flat = np.array(axes).flatten()

    for idx, key in enumerate(task_keys):
        ax = axes_flat[idx]
        task = key_to_display(key)
        env_id = key.rsplit("/", 1)[0]
        has_data = False

        for name, config in RUN_CONFIGS.items():
            color = config["color"]
            # Build one smoothed curve per seed (stitching that seed's runs).
            seed_curves = []
            for seed_paths in method_seeds[name]:
                seed_dfs = [run_hist.get(p) for p in seed_paths]
                res = concat_seed(seed_dfs, key)
                if res is None:
                    continue
                steps, values = res
                values = relative(values, env_id)
                if X_MAX is not None:
                    mask = steps <= X_MAX
                    steps, values = steps[mask], values[mask]
                if len(steps) == 0:
                    continue
                smoothed, smooth_idx = smooth(values)
                seed_curves.append((steps[smooth_idx], smoothed))

            if not seed_curves:
                print(f"  [{name}] no data for: {key}")
                continue
            has_data = True

            if SHOW_INDIVIDUAL_SEEDS:
                for s_steps, s_values in seed_curves:
                    ax.plot(s_steps, s_values, color=color, alpha=0.25, linewidth=0.8)

            agg = aggregate_seeds(seed_curves)
            if agg is not None:
                grid, mean, std = agg
                ax.fill_between(grid, mean - std, mean + std, color=color,
                                alpha=0.2, linewidth=0)
                ax.plot(grid, mean, color=color, linewidth=2.0, label=name)
            else:
                s_steps, s_values = seed_curves[0]
                ax.plot(s_steps, s_values, color=color, linewidth=2.0, label=name)

        if not has_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

        ax.set_title(task, fontsize=20)
        if X_MAX is not None:
            ax.set_xlim(0, X_MAX)
        # Keep the autoscaled top, but never below Y_MAX_MIN: a task stuck near zero
        # would otherwise autoscale to a sub-1% range and print nothing but 0% ticks.
        ax.set_ylim(0, max(ax.get_ylim()[1], Y_MAX_MIN))
        ax.set_xlim(left=0)
        # Values are already on a 0-100 relative scale, so xmax=100 just appends '%'.
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(axis="both", pad=8)
        ax.grid(True, alpha=0.3)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Environment steps", labelpad=15)

    for idx in range(n_plots, nrows * ncols):
        axes_flat[idx].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=28, framealpha=0.9, ncol=1)

    plt.tight_layout(h_pad=0.5, w_pad=0.5)
    fig.text(0.0, 0.5, "Relative Performance (%)", va="center", ha="center",
             rotation="vertical", fontsize=28)

    os.makedirs(os.path.join(ROOT_DIR, "data", "figures"), exist_ok=True)
    out_png = os.path.join(ROOT_DIR, "data", "figures", "mt_algos_training_curves.png")
    out_svg = os.path.join(ROOT_DIR, "data", "figures", "mt_algos_training_curves.svg")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.savefig(out_svg, bbox_inches="tight")
    print(f"Saved to {out_png}")
    plt.close()


if __name__ == "__main__":
    create_plot()
