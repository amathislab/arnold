import _srcpath  # noqa: F401
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter

from analysis.performance import expert_means, task_names


def main(default_panel=None, default_relative=True, default_window=5, default_smoothing="moving", default_combine=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", nargs="+", type=Path, default=[Path("data/analysis/learning_curves.csv.gz")])
    parser.add_argument("--panel", default=default_panel)
    parser.add_argument("--window", type=int, default=default_window)
    parser.add_argument("--smoothing", choices=["moving", "savgol"], default=default_smoothing)
    parser.add_argument("--relative", dest="relative", action="store_true")
    parser.add_argument("--raw", dest="relative", action="store_false")
    parser.set_defaults(relative=default_relative)
    parser.add_argument("--combine_tasks", action="store_true", default=default_combine)
    parser.add_argument("--offsets", nargs="*", default=[], help="Stage step offsets, e.g. transfer=50000000")
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/learning_curves"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.curves], ignore_index=True)
    if args.panel is not None:
        data = data[data.panel == args.panel]
    if data.empty:
        raise ValueError(f"No curves for panel {args.panel}")
    offsets = {stage: float(value) for stage, value in (item.split("=", 1) for item in args.offsets)}
    data["plot_step"] = data.step - data.stage.map(offsets).fillna(0)
    names = task_names()
    experts = expert_means() if args.relative and (data.metric == "solved_fraction").any() else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for (panel, task, metric), subset in data.groupby(["panel", "task", "metric"]):
        key = (panel, "combined" if args.combine_tasks else task, metric)
        if key not in figures:
            figures[key] = plt.subplots(figsize=(7, 4))
        fig, ax = figures[key]
        for (method, stage), group in subset.groupby(["method", "stage"]):
            curves = []
            for seed, values in group.groupby("seed"):
                values = values.sort_values("plot_step")
                if args.smoothing == "savgol":
                    if len(values) < args.window or args.window < 3 or args.window % 2 != 1:
                        raise ValueError("Savitzky-Golay window must be odd, >=3, and no longer than each curve")
                    y = pd.Series(savgol_filter(values.value, args.window, 2), index=values.index)
                else:
                    y = values.value.rolling(args.window, min_periods=args.window).mean()
                if metric == "solved_fraction" and args.relative:
                    y = y / experts[task] * 100
                series = pd.Series(y.to_numpy(), index=values.plot_step.to_numpy(), name=seed).dropna()
                curves.append(series)
            aligned = pd.concat(curves, axis=1).dropna()
            if aligned.empty:
                raise ValueError(f"No shared steps after smoothing: {panel}, {task}, {method}, {stage}")
            mean = aligned.mean(axis=1)
            line, = ax.plot(mean.index, mean, label=f"{names[task]}, {method}" if args.combine_tasks else f"{method}, {stage}")
            if len(curves) > 1:
                std = aligned.std(axis=1, ddof=0)
                ax.fill_between(mean.index, mean-std, mean+std, color=line.get_color(), alpha=.15)
    for (panel, task, metric), (fig, ax) in figures.items():
        if metric == "solved_fraction" and args.relative:
            ax.axhline(100, color="gray", linestyle="--")
        ylabel = "Relative performance (%)" if metric == "solved_fraction" and args.relative else "Solved fraction" if metric == "solved_fraction" else "Episode reward"
        ax.set(title="" if task == "combined" else names[task], xlabel="Training steps", ylabel=ylabel)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{panel}_{task}_{metric}.svg")
        fig.savefig(args.out_dir / f"{panel}_{task}_{metric}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    main()
