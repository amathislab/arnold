"""Compare gait-normalized policy activity with nine subjects' human EMG."""
import _srcpath  # noqa: F401
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.emg import correlations, leave_one_out, policy_statistics
from analysis.gait import resample


POLICIES = ("Arnold", "OBC", "Kinesis")
COLORS = ("#2580b8", "#209688", "#54a24b")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/analysis/emg"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/emg"))
    args = parser.parse_args()
    with np.load(args.data_dir / "human.npz") as data:
        human, subjects, muscles = data["profiles"], data["subjects"], data["muscles"]
    with np.load(args.data_dir / "simulation.npz") as data:
        simulated_muscles = list(data["emg_muscles"])
        order = [simulated_muscles.index(muscle) for muscle in muscles]
        profiles = {policy: data[policy.lower() + "_left"][order] for policy in POLICIES}
    values = {policy: correlations(human, profile) for policy, profile in profiles.items()}
    values["Human"] = leave_one_out(human)
    means = {policy: value.mean(axis=1) for policy, value in values.items()}
    comparisons = policy_statistics({policy: means[policy] for policy in POLICIES})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"policy": policy, "subject": int(subject), "muscle": str(muscle), "r": float(value)}
                  for policy, matrix in values.items() for subject, row in zip(subjects, matrix)
                  for muscle, value in zip(muscles, row)]).to_csv(args.out_dir / "correlations.csv", index=False)
    pd.DataFrame([{"policy": policy, "subject": int(subject), "r": float(value)}
                  for policy, row in means.items() for subject, value in zip(subjects, row)]).to_csv(
        args.out_dir / "subject_means.csv", index=False)
    comparisons.to_csv(args.out_dir / "comparisons.csv", index=False)
    plt.rcParams["svg.fonttype"] = "none"
    figure, axis = plt.subplots(figsize=(5, 3.5))
    labels = ("Human",) + POLICIES
    points = [means[policy] for policy in labels]
    axis.bar(np.arange(4), [v.mean() for v in points],
             yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in points],
             color=("#9c9c9c",) + COLORS, capsize=3)
    for index, row in enumerate(points):
        axis.scatter(index + np.linspace(-0.08, 0.08, len(row)), row, color="black", s=12)
    axis.set_xticks(np.arange(4))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Mean correlation across muscles")
    height = max(v.max() for v in points) + 0.04
    for row in comparisons.itertuples():
        if row.p_bonferroni < 0.05:
            first, second = labels.index(row.a), labels.index(row.b)
            axis.plot([first, first, second, second], [height - 0.01, height, height, height - 0.01], color="black")
            axis.text((first + second) / 2, height, "*", ha="center", va="bottom")
            height += 0.07
    axis.set_ylim(top=height + 0.04)
    figure.tight_layout()
    figure.savefig(args.out_dir / "correlations.svg")
    plt.close(figure)
    figure, axes = plt.subplots(1, 4, figsize=(12, 2.8))
    for axis, muscle in zip(axes, (0, 1, 2, 5)):
        for subject in human:
            values = subject[:, muscle]
            axis.plot(np.linspace(0, 100, len(values)), values / values.max(), color="gray", alpha=0.3)
        values = human[:, :, muscle].mean(axis=0)
        axis.plot(np.linspace(0, 100, len(values)), values / values.max(), color="black", label="Human")
        for policy, color in zip(POLICIES, COLORS):
            values = resample(profiles[policy].T, human.shape[1])[:, muscle]
            axis.plot(np.linspace(0, 100, len(values)), values / values.max(), color=color, label=policy)
        axis.set(title=str(muscles[muscle]), xlabel="Gait cycle (%)", ylabel="Normalized activity")
    axes[-1].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(args.out_dir / "profiles.svg")
    plt.close(figure)
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
