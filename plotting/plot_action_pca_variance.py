import _srcpath  # noqa: F401
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.signals import selection_paths
from analysis.subspaces import load_signals, principal_components


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/analysis/signals"))
    parser.add_argument("--activations_dir", type=Path,
                        help="Existing per-episode recordings for one policy")
    parser.add_argument("--selections", default="data/analysis/reproduction/signals.json")
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/cumulative_variance"))
    args = parser.parse_args()
    policies = {}
    if args.activations_dir:
        tasks = sorted({p.stem.rsplit("_episode_", 1)[0]
                        for p in args.activations_dir.glob("*_episode_*.h5")})
        if not tasks:
            parser.error("--activations_dir contains no task_episode_N.h5 files")
        selections = [(args.activations_dir.name, task, args.activations_dir / (task + ".h5"))
                      for task in tasks]
    else:
        selections = selection_paths(args.data_dir, "action_pca", args.selections)
    for policy, task, path in selections:
        policies.setdefault(policy, {})[task] = load_signals(path, "action_means")[0]
    rows = []
    for policy, tasks in policies.items():
        _, combined = principal_components(np.cov(np.vstack(list(tasks.values())).T))
        for task, values in tasks.items():
            covariance = np.cov(values.T)
            eigenvalues, _ = principal_components(covariance)
            for scope, variances in [("task", eigenvalues), ("global", np.diag(combined @ covariance @ combined.T))]:
                for dimension, ev in enumerate(np.cumsum(variances) / np.trace(covariance), 1):
                    rows.append(dict(policy=policy, task=task, scope=scope, dimension=dimension, explained_variance=ev))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(rows)
    data.to_csv(args.out_dir / "variance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    for (policy, scope), group in data.groupby(["policy", "scope"]):
        mean = group.groupby("dimension").explained_variance.mean()
        ax.plot(mean.index, mean, label=f"{policy}, {scope}")
    ax.set(xlabel="Components", ylabel="Cumulative explained variance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "variance.svg")
    fig.savefig(args.out_dir / "variance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
