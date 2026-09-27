import _srcpath  # noqa: F401
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, wilcoxon

from analysis.performance import REPO_ROOT, relative_matrix, task_names, mean_sem, holm, rank_biserial, expert_means

METHODS = {"arnold": "Arnold", "mt_sac": "MT-SAC", "mt_ppo": "MT-PPO", "ppo_t": "MT-PPO+T",
           "ppo_t_sv": "MT-PPO+T+SV", "bc": "BC", "obc_task_sv": "OBC + Task-SV",
           "obc_wo_obs_norm": "OBC w/o Obs.Norm", "obc_ppo": "OBC-PPO", "obc": "OBC"}
FAMILIES = {"Finger reach": ["hand_index_reach", "hand_little_reach", "hand_middle_reach", "hand_ring_reach", "hand_thumb_reach"],
            "Elbow pose": ["elbow_pose"], "Baoding": ["baoding_p1_cw", "baoding_p1_ccw", "baoding_p2", "baoding_p2_overlap"],
            "Reorient": ["reorient", "pen"], "Relocate": ["relocate"], "Walk to point": ["kinesis"]}


def write_latex(path, data, rows, columns, significant=()):
    lines = ["Method & " + " & ".join(columns) + " " + chr(92)*2]
    for method in rows:
        values = []
        for column in columns:
            cell = data[(data.method == method) & (data.column == column)].iloc[0]
            text = f"{cell['mean']:.1f}"
            if np.isfinite(cell['sem']):
                text += rf" \pm {cell['sem']:.1f}"
            if column == "Avg" and method in significant:
                text += "^{*}"
            values.append("$" + text + "$")
        lines.append(METHODS[method] + " & " + " & ".join(values) + " " + chr(92)*2)
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out_dir", type=Path, default=Path("data/analysis/tables"))
    args = parser.parse_args()
    names = task_names()
    tasks = [task for family in FAMILIES.values() for task in family]
    matrices = {method: relative_matrix(method, tasks, args.root) for method in METHODS}
    family_rows, task_rows, stats_rows = [], [], []
    for method, matrix in matrices.items():
        for family, subset in list(FAMILIES.items()) + [("Avg", tasks)]:
            values = matrix[:, [tasks.index(task) for task in subset]].mean(axis=1)
            mean, sem = mean_sem(values)
            family_rows.append(dict(method=method, column=family, mean=mean, sem=sem, n_seeds=len(matrix)))
        for index, task in enumerate(tasks):
            mean, sem = mean_sem(matrix[:, index])
            task_rows.append(dict(method=method, column=names[task], task=task, mean=mean, sem=sem, n_seeds=len(matrix)))
        if method != "arnold":
            reference = matrices["arnold"].mean(axis=0)
            values = matrix.mean(axis=0)
            difference = reference-values
            test = wilcoxon(reference, values, alternative="two-sided", zero_method="wilcox") if np.any(difference) else None
            stats_rows.append(dict(method=method, W=float(test.statistic) if test else 0.,
                                   p=float(test.pvalue) if test else 1., n_tasks=len(tasks),
                                   n_nonzero=np.count_nonzero(difference), rank_biserial=rank_biserial(difference)))
    statistics = pd.DataFrame(stats_rows)
    statistics["p_holm"] = holm(statistics.p)
    families, per_task = pd.DataFrame(family_rows), pd.DataFrame(task_rows)
    expert_rows = []
    arnold = matrices["arnold"]
    expert_values = expert_means(args.root)
    for index, task in enumerate(tasks):
        values = arnold[:, index]-100
        mean, sem = mean_sem(values)
        test = ttest_1samp(values, 0, alternative="greater") if np.any(values) else None
        p = test.pvalue if test else 1.
        arnold_mean, arnold_sem = mean_sem(arnold[:, index] * expert_values[task] / 100)
        expert_rows.append(dict(task=task, expert=expert_values[task], arnold=arnold_mean, arnold_sem=arnold_sem,
                               improvement=mean, sem=sem, t=float(test.statistic) if test else 0., p=p, n_seeds=len(arnold)))
    expert = pd.DataFrame(expert_rows)
    expert["p_holm"] = holm(expert.p)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in [("families", families), ("per_task", per_task), ("significance", statistics), ("expert_improvement", expert)]:
        data.to_csv(args.out_dir / (filename+".csv"), index=False)
    significant = statistics.loc[statistics.p_holm < .05, "method"]
    write_latex(args.out_dir / "families.tex", families, METHODS, list(FAMILIES) + ["Avg"], significant)
    write_latex(args.out_dir / "per_task.tex", per_task, METHODS, [names[task] for task in tasks])
    lines = ["Task & Expert & Arnold & Improvement (\\%) & t & p (Holm) " + chr(92)*2]
    for row in expert.itertuples():
        lines.append(f"{names[row.task]} & {row.expert:.3f} & ${row.arnold:.3f} " + chr(92) + f"pm {row.arnold_sem:.3f}$ & ${row.improvement:+.2f} " + chr(92) + f"pm {row.sem:.2f}$ & {row.t:.3f} & {row.p_holm:.4g} " + chr(92)*2)
    (args.out_dir / "expert_improvement.tex").write_text("\n".join(lines)+"\n")
    lines = ["Method & W & p & p (Holm) & Rank-biserial " + chr(92)*2]
    for row in statistics.itertuples():
        lines.append(f"{METHODS[row.method]} & {row.W:g} & {row.p:.4g} & {row.p_holm:.4g} & {row.rank_biserial:.3f} " + chr(92)*2)
    (args.out_dir / "significance.tex").write_text("\n".join(lines)+"\n")
    print(families[families.column == "Avg"].to_string(index=False))


if __name__ == "__main__":
    main()
