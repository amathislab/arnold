import _srcpath  # noqa: F401
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.performance import load_benchmarks, task_names, expert_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/capacity"))
    args = parser.parse_args()
    names, experts = task_names(), expert_means()
    rows = []
    for method in ["obc", "obc_m", "obc_s", "obc_xs"]:
        result = load_benchmarks(method)[0]
        for task in names:
            entry = result[task]
            horizon = entry["max_episode_steps"]
            sem = entry["std_solved_steps"] / horizon / np.sqrt(entry["num_episodes"])
            rows.append(dict(method=method, task=task, relative_performance=entry["avg_solved_step_frac"] / experts[task]*100,
                             sem=sem/experts[task]*100, n_episodes=entry["num_episodes"]))
    data = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.out_dir / "performance.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    for index, (method, group) in enumerate(data.groupby("method", sort=False)):
        ax.bar(np.arange(len(names)) + (index-1.5)*.2, group.relative_performance, width=.2, yerr=group['sem'], label=method)
    ax.axhline(100, color="gray", linestyle="--")
    ax.set(xticks=np.arange(len(names)), xticklabels=list(names.values()), ylabel="Relative performance (%)")
    ax.tick_params(axis="x", labelrotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_dir / "capacity.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
