"""Fit four varimax-rotated gait factors and plot temporal and muscle loadings."""
import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.gait_factors import bartlett_sphericity, kmo_measure, run_varimax_pca


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=Path("data/analysis/emg"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/figures/gait_factors"))
    args = parser.parse_args()
    with np.load(args.data_dir / "simulation.npz") as data:
        activity, muscles = data["arnold_right"], data["factor_muscles"]
    factors, loadings, retained, pca, _ = run_varimax_pca(activity)
    chi2, pvalue = bartlett_sphericity(activity.T)
    stats = {"n_factors": int(retained), "n_muscles": len(muscles), "n_time": activity.shape[1],
             "bartlett_chi2": chi2, "bartlett_df": len(muscles) * (len(muscles) - 1) // 2,
             "bartlett_p": pvalue, "kmo": kmo_measure(activity.T),
             "explained_variance": float(pca.explained_variance_ratio_[:retained].sum())}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "factors.npy", factors)
    np.save(args.out_dir / "loadings.npy", loadings)
    pd.DataFrame(loadings, index=muscles, columns=[f"F{i + 1}" for i in range(retained)]).to_csv(
        args.out_dir / "loadings.csv", index_label="muscle")
    (args.out_dir / "statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    plt.rcParams["svg.fonttype"] = "none"
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    for index, factor in enumerate(factors):
        axes[0].plot(np.linspace(0, 100, len(factor)), factor, label=f"F{index + 1}")
    axes[0].set(xlabel="Gait cycle (%)", ylabel="Factor activity")
    axes[0].legend()
    image = axes[1].imshow(loadings.T, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_xticks(np.arange(len(muscles)))
    axes[1].set_xticklabels(muscles, rotation=90)
    axes[1].set_yticks(np.arange(retained))
    axes[1].set_yticklabels([f"F{i + 1}" for i in range(retained)])
    figure.colorbar(image, ax=axes[1], label="Loading")
    figure.tight_layout()
    figure.savefig(args.out_dir / "factors.svg")
    plt.close(figure)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
