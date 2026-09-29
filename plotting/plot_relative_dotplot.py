import _srcpath  # noqa: F401
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.performance import load_benchmarks, task_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="obc_task_sv")
    parser.add_argument("--reference", default="obc")
    parser.add_argument("--output", type=Path, default=Path("data/figures/relative_dotplot.svg"))
    args = parser.parse_args()
    names = task_names()
    tasks = list(names)
    def matrix(method):
        return np.asarray([[result[task]["avg_solved_step_frac"] for task in tasks] for result in load_benchmarks(method)])
    differences = matrix(args.method) / matrix(args.reference).mean(axis=0)*100-100
    order = np.argsort(differences.mean(axis=0))
    fig, ax = plt.subplots(figsize=(8, 6))
    for seed in differences:
        ax.scatter(seed[order], np.arange(len(tasks)), facecolors="none", edgecolors="black", s=20)
    ax.errorbar(differences.mean(axis=0)[order], np.arange(len(tasks)),
                xerr=differences.std(axis=0, ddof=1)[order], fmt="o", color="black", capsize=3)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(yticks=np.arange(len(tasks)), yticklabels=[names[tasks[i]] for i in order],
           xlabel=f"Performance change relative to {args.reference} (%)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
