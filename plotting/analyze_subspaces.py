import _srcpath  # noqa: F401
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.signals import selection_paths
from analysis.subspaces import load_signals, principal_components, compare_subspaces


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/analysis/signals"))
    parser.add_argument("--selection", choices=["subspaces", "capacity_recordings"], default="subspaces")
    parser.add_argument("--selections", default="data/analysis/reproduction/signals.json")
    parser.add_argument("--successful", type=int)
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/subspaces"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells = {}
    for policy, task, path in selection_paths(args.data_dir, args.selection, args.selections):
        values, count = load_signals(path, "actions", args.successful)
        covariance = np.cov(values.T)
        _, components = principal_components(covariance)
        cells[policy, task] = (covariance, components, count)
    rows = []
    policies = list(dict.fromkeys(policy for policy, task in cells))
    keys = [(policy, task) for policy in policies for task in sorted(task for p, task in cells if p == policy)]
    for left, right in combinations(keys, 2):
        kind = "across_tasks" if left[0] == right[0] else "across_policies" if left[1] == right[1] else None
        if kind is None:
            continue
        covariance, components, count = cells[left]
        _, other, other_count = cells[right]
        for dimension in range(1, len(components) + 1):
            pvd, pad = compare_subspaces(covariance, components, other, dimension)
            rows.append(dict(comparison=kind, policy_a=left[0], task_a=left[1],
                             policy_b=right[0], task_b=right[1], dimension=dimension,
                             pvd=pvd, pad=pad, n_a=count, n_b=other_count))
    pairs = pd.DataFrame(rows)
    pairs.to_csv(args.out_dir / "pairs.csv", index=False)
    pairs["group"] = np.where(pairs.comparison == "across_tasks", pairs.policy_a,
                              pairs.policy_a + " / " + pairs.policy_b)
    summary = pairs.groupby(["comparison", "group", "dimension"])[["pvd", "pad"]].agg(["mean", lambda values: values.std(ddof=0), "count"])
    summary.columns = ["_".join(column).replace("<lambda_0>", "std") for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    for kind, subset in summary.groupby("comparison"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for label, curve in subset.groupby("group"):
            for ax, metric in zip(axes, ["pvd", "pad"]):
                x = curve.dimension.to_numpy()
                y, sd = curve[metric + "_mean"].to_numpy(), curve[metric + "_std"].to_numpy()
                ax.plot(x, y, label=label)
                ax.fill_between(x, y - sd, y + sd, alpha=.15)
                ax.set(xlabel="Components", ylabel="PVD" if metric == "pvd" else "PAD (degrees)")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out_dir / (kind + ".svg"))
        plt.close(fig)


if __name__ == "__main__":
    main()
