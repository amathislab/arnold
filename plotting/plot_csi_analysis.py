import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.performance import expert_means, task_names


TASK_NAME_MAPPING = task_names()


def summarize_runs(values):
    return dict(mean=float(np.mean(values)) if len(values) else float("nan"),
                sem=float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else float("nan"),
                n_runs=len(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("data/final_benchmarks_extra"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/csi_analysis"))
    args = parser.parse_args()
    names, experts = task_names(), expert_means()
    groups = {"frozen": ("csi_notrain_server", ["111_gpu_csi_all_bc_student"]),
              "RL": ("csi_server", ["333", "444", "555"]),
              "OBC": ("csi_bc_server", ["666", "777", "888"])}
    measurements, summaries = [], []
    for task in names:
        dimensions = [1, 2, 5] if task == "elbow_pose" else [1, 2, 5, 10, 20] if task.startswith("hand_") else [1, 2, 5, 10, 20, 30, 40]
        for method, (directory, runs) in groups.items():
            for dimension in dimensions:
                values = []
                for run in runs:
                    candidates = sorted((args.results / directory / f"csi{dimension}_all").glob(f"{run}_{task}*_csi_{dimension}_results.json"))
                    # The task key distinguishes Baoding P2 from P2 overlap.
                    entries = [(file, json.loads(file.read_text())) for file in candidates]
                    entries = [(file, data[task]) for file, data in entries if task in data]
                    if not entries:
                        continue
                    file, entry = max(entries, key=lambda pair: (int(re.search(r"model_(\d+)_steps", pair[0].name).group(1)), pair[0].name))
                    value = entry["avg_solved_step_frac"] / experts[task] * 100
                    values.append(value)
                    measurements.append(dict(task=task, method=method, dimension=dimension,
                                             run=runs.index(run), relative_performance=value, n_episodes=entry["num_episodes"]))
                summaries.append(dict(task=task, method=method, dimension=dimension, **summarize_runs(values)))
    data, summary = pd.DataFrame(measurements), pd.DataFrame(summaries)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_dir / "runs.csv", index=False)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    fig, axes = plt.subplots(4, 4, figsize=(14, 12))
    for ax, (task, name) in zip(axes.flat, names.items()):
        for method, group in summary[summary.task == task].groupby("method", sort=False):
            line, = ax.plot(group.dimension, group['mean'], marker="o", label=method)
            ax.fill_between(group.dimension, group['mean']-group['sem'], group['mean']+group['sem'], alpha=.15, color=line.get_color())
        ax.axhline(100, color="gray", linestyle="--")
        ax.set(title=name, xlabel="Control dimensions", ylabel="Relative performance (%)")
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "csi.svg")
    fig.savefig(args.out_dir / "csi.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
