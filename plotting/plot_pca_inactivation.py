import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", nargs="+", type=Path, default=[Path("data/analysis/historical_curves.csv")])
    parser.add_argument("--experts", type=Path, default=Path("data/final_benchmarks/expert_policies"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/pca_inactivation"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.curves], ignore_index=True)
    experts = {}
    for task in data.task.unique():
        result = json.loads((args.experts / f"expert_{task}_results.json").read_text())[task]
        experts[task] = result["avg_solved_step_frac"]
    data["relative_performance"] = data.solved_fraction / data.task.map(experts) * 100
    data["relative_sem"] = data.solved_fraction_sem / data.task.map(experts) * 100
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_dir / "relative_curves.csv", index=False)
    for method, subset in data.groupby("method"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for (scope, basis), group in subset.groupby(["scope", "basis_policy"]):
            label = f"{scope}, {basis}"
            color = "#8172B3" if basis == "expert" else "#4C72B0" if scope == "task" else "#C44E52"
            for task, curve in group.groupby("task"):
                curve = curve.sort_values("dimension")
                axes[0].plot(curve.dimension, curve.relative_performance, color=color, alpha=.12, linewidth=1)
            summary = group.groupby("dimension").agg(performance=("relative_performance", "mean"), reconstruction=("reconstruction", "mean"))
            axes[0].plot(summary.index, summary.performance, label=label, color=color)
            axes[1].plot(summary.index, summary.reconstruction, label=label, color=color)
        axes[0].set(ylabel="Relative performance (%)", xlabel="Components")
        axes[1].set(ylabel="Explained variance" if method == "pca" else "Reconstruction R²", xlabel="Components")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / (method + ".svg"))
        fig.savefig(args.out_dir / (method + ".png"))
        plt.close(fig)
        for task, task_data in subset.groupby("task"):
            fig, ax = plt.subplots(figsize=(5, 4))
            ev = ax.twinx()
            for (scope, basis), curve in task_data.groupby(["scope", "basis_policy"]):
                curve = curve.sort_values("dimension")
                x, y, sem = (curve[column].to_numpy() for column in ["dimension", "relative_performance", "relative_sem"])
                line, = ax.plot(x, y, label=f"{scope}, {basis}")
                ax.fill_between(x, y-sem, y+sem, alpha=.15, color=line.get_color())
                ev.plot(x, curve.reconstruction, linestyle="--", color=line.get_color())
            ax.set(title=task, xlabel="Components", ylabel="Relative performance (%)")
            ev.set_ylabel("Explained variance" if method == "pca" else "Reconstruction R²")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(args.out_dir / f"{method}_{task}.svg")
            fig.savefig(args.out_dir / f"{method}_{task}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
