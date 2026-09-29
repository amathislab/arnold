import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.performance import task_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("data/final_benchmarks_extra/bilateral/bilateral.json"))
    parser.add_argument("--output", type=Path, default=Path("data/figures/bilateral_reward.svg"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    tasks = list(next(iter(data.values())))
    width = .8/len(data)
    fig, ax = plt.subplots(figsize=(9, 4))
    for index, (method, results) in enumerate(data.items()):
        ax.bar(np.arange(len(tasks)) + (index-(len(data)-1)/2)*width,
               [results[task]["avg_cum_reward"] for task in tasks], width=width, label=method)
    names = task_names()
    ax.set(xticks=np.arange(len(tasks)), xticklabels=[names[task] for task in tasks], ylabel="Episode reward")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
