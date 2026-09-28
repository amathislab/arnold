"""Baoding CCW joint-position and joint-velocity PCA and human comparison."""
import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.kinematics import THRESHOLDS, dimensionality, pooled_kinematics
from analysis.signals import iter_episodes, selection_paths


LABELS = ("Expert", "OBC", "Arnold")
COLORS = ("#9c9c9c", "#209688", "#2580b8")


def plot_results(summary, curves, reference, policies, tasks, output):
    plt.rcParams["svg.fonttype"] = "none"
    for task in tasks:
        figure, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for axis, quantity in zip(axes[:2], ("position", "velocity")):
            for policy, label, color in zip(policies, LABELS, COLORS):
                values = curves[(curves.task == task) & (curves.policy == policy)
                                & (curves.quantity == quantity)]
                axis.step(values.component, values.cumulative_variance,
                          where="mid", label=label, color=color)
            for threshold in THRESHOLDS:
                axis.axhline(threshold, color="black", linestyle="--", linewidth=0.6)
            axis.set(title=f"Joint {quantity}", xlabel="Number of PCs",
                     ylabel="Cumulative explained variance", ylim=(0, 1.02))
            axis.legend(fontsize=8)
        x = np.arange(2)
        for index, (policy, label, color) in enumerate(zip(policies, LABELS, COLORS)):
            values = summary[(summary.task == task) & (summary.policy == policy)].set_index("quantity")
            axes[2].bar(x + (index - 1.5) * 0.2,
                        values.loc[["position", "velocity"], "avg_dim"],
                        width=0.2, label=f"{label} (23 joints)", color=color)
        human = reference.set_index("quantity")
        human_means = human.loc[["position", "velocity"], ["dim_85", "dim_95"]].mean(axis=1)
        axes[2].bar(x + 0.3, human_means, width=0.2, color="#beada1", label="Human (20 joints)")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(["Position", "Velocity"])
        axes[2].set_ylabel("Mean PC count at 85% and 95%")
        axes[2].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(output / f"{task}.svg", bbox_inches="tight")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/activations"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/baoding_pca"))
    parser.add_argument("--selections", type=Path, default=Path("data/analysis/reproduction/signals.json"))
    parser.add_argument("--human_reference", type=Path, default=Path("data/analysis/reproduction/baoding_human.csv"))
    args = parser.parse_args()
    selections = json.loads(args.selections.read_text())
    policies = selections["baoding_pca"]["policies"]
    tasks = selections["baoding_pca"]["tasks"]
    rows, curves = [], []
    for policy, task, path in selection_paths(args.data_dir, "baoding_pca", args.selections):
        trajectories = [signals["joint_positions"].astype(np.float64)
                        for _, signals in iter_episodes(path, ["joint_positions"])]
        if len(trajectories) != selections["episodes"]:
            raise ValueError(f"{policy}/{task}: expected {selections['episodes']} episodes, found {len(trajectories)}")
        for quantity in ("position", "velocity"):
            samples = pooled_kinematics(trajectories, quantity)
            counts, curve = dimensionality(samples)
            rows.append({"policy": policy, "task": task, "quantity": quantity,
                         "dim_85": counts[0], "dim_95": counts[1], "avg_dim": float(np.mean(counts)),
                         "n_episodes": len(trajectories), "n_timesteps": len(samples),
                         "n_joints": samples.shape[1]})
            curves.extend({"policy": policy, "task": task, "quantity": quantity,
                           "component": index, "cumulative_variance": float(value)}
                          for index, value in enumerate(curve, 1))
    summary, curves = pd.DataFrame(rows), pd.DataFrame(curves)
    reference = pd.read_csv(args.human_reference)
    reference["avg_dim"] = reference[["dim_85", "dim_95"]].mean(axis=1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "dimensionality.csv", index=False)
    curves.to_csv(args.out_dir / "cumulative_variance.csv", index=False)
    reference.to_csv(args.out_dir / "human_comparison.csv", index=False)
    plot_results(summary, curves, reference, policies, tasks, args.out_dir)
    print(summary.to_string(index=False))
    print(f"Wrote PCA results and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
