"""Compute and plot the manuscript's 11-task hand smoothness comparison."""
import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.signals import iter_episodes, selection_paths
from analysis.smoothness import METRICS, compute_metrics, paired_statistics


LABELS = ("Expert", "Single-task OBC", "Multi-task OBC")
COLORS = ("#9c9c9c", "#9467bd", "#209688")
TITLES = (r"Mean $|\Delta a|$ ↓", "SPARC ↑", "LDLJ ↑", "Velocity peaks ↓")


def plot_smoothness(per_task, comparisons, policies, tasks, output):
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(1, 4, figsize=(13.2, 3.2))
    x = np.arange(len(policies))
    jitter = (np.random.RandomState(0).rand(len(tasks)) - 0.5) * 0.18
    for axis, metric, title in zip(axes, METRICS, TITLES):
        values = per_task[metric].unstack("policy").reindex(index=tasks, columns=policies)
        means, sems = values.mean(), values.sem()
        axis.bar(x, means, yerr=sems, color=COLORS, capsize=3, width=0.6)
        for offset, row in zip(jitter, values.to_numpy()):
            axis.plot(x + offset, row, color="black", alpha=0.25, linewidth=0.3)
            axis.scatter(x + offset, row, color="black", s=10, alpha=0.5)
        low = min(values.to_numpy().min(), (means - sems).min())
        high = max(values.to_numpy().max(), (means + sems).max())
        span = high - low or 1.0
        height = high + 0.08 * span
        for row in comparisons[comparisons.metric == metric].itertuples():
            if row.p_holm >= 0.05:
                continue
            first, second = policies.index(row.a), policies.index(row.b)
            axis.plot([first, first, second, second],
                      [height - 0.02 * span, height, height, height - 0.02 * span],
                      color="black", linewidth=0.7)
            axis.text((first + second) / 2, height, "**" if row.p_holm < 0.01 else "*",
                      ha="center", va="bottom", fontsize=9)
            height += 0.14 * span
        axis.set_ylim(top=height + 0.08 * span)
        axis.set_xticks(x)
        axis.set_xticklabels(LABELS, rotation=20, ha="right", fontsize=8)
        axis.set_title(title)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/activations"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/smoothness"))
    parser.add_argument("--selections", type=Path, default=Path("data/analysis/reproduction/signals.json"))
    args = parser.parse_args()
    selections = json.loads(args.selections.read_text())
    policies = selections["smoothness"]["policies"]
    tasks = selections["smoothness"].get("tasks", selections["tasks"])
    rows = []
    for policy, task, path in selection_paths(args.data_dir, "smoothness", args.selections):
        count = 0
        for episode, signals in iter_episodes(path, ["actions", "joint_positions"]):
            rows.append({"policy": policy, "task": task, "episode": episode,
                         "episode_length": len(signals["actions"]),
                         **compute_metrics(signals["actions"], signals["joint_positions"])})
            count += 1
        if count != selections["episodes"]:
            raise ValueError(f"{policy}/{task}: expected {selections['episodes']} episodes, found {count}")
    per_episode = pd.DataFrame(rows)
    if not np.isfinite(per_episode[list(METRICS)].to_numpy()).all():
        raise ValueError("Smoothness metrics contain undefined episode values")
    per_task = per_episode.groupby(["policy", "task"])[list(METRICS)].mean()
    per_task["episodes"] = per_episode.groupby(["policy", "task"]).size()
    comparisons = paired_statistics(per_task, policies, tasks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_episode.to_csv(args.out_dir / "per_episode.csv", index=False)
    per_task.to_csv(args.out_dir / "per_task.csv")
    comparisons.to_csv(args.out_dir / "comparisons.csv", index=False)
    plot_smoothness(per_task, comparisons, policies, tasks, args.out_dir / "smoothness.svg")
    print(f"Wrote {len(per_episode)} episodes, {len(per_task)} task means and {len(comparisons)} comparisons to {args.out_dir}")


if __name__ == "__main__":
    main()
