import _srcpath  # noqa: F401  # adds ../src to sys.path (see _srcpath.py)
import glob
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from definitions import ROOT_DIR


def holm_bonferroni(pvalues):
    """Holm-Bonferroni step-down correction.

    Returns family-wise-error-controlled adjusted p-values in the same order as
    the input. Controls the same family-wise error rate as plain Bonferroni but
    is uniformly more powerful: the smallest raw p is multiplied by k, the next
    by k-1, ..., the largest by 1, with monotonicity enforced across the sorted
    sequence.
    """
    pvalues = np.asarray(pvalues, dtype=float)
    k = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(k)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, min((k - rank) * pvalues[idx], 1.0))
        adjusted[idx] = running_max
    return adjusted

# Mapping for better task display names
TASK_NAME_MAPPING = {
    "hand_little_reach": "Little\n reach",
    "hand_index_reach": "Index\n reach",
    "hand_middle_reach": "Middle\n reach",
    "hand_ring_reach": "Ring\n reach",
    "hand_thumb_reach": "Thumb\n reach",
    "reorient": "Die\n reorient",
    "pen": "Pen\n reorient",
    "baoding_p1_cw": "Baoding\n CW",
    "baoding_p1_ccw": "Baoding\n CCW",
    "baoding_p2_overlap": "Baoding\n hard",
    "baoding_p2": "Baoding\n harder",
    "elbow_pose": "Elbow\n pose",
    "relocate": "Object\n relocation",
    "kinesis": "Walk to\n point",
}

# Metric to use for each task
TASK_METRIC_MAP = {
    "hand_little_reach": "solved_steps",
    "hand_index_reach": "solved_steps",
    "hand_middle_reach": "solved_steps",
    "hand_ring_reach": "solved_steps",
    "hand_thumb_reach": "solved_steps",
    "reorient": "solved",
    "pen": "solved_steps",
    "baoding_p1_cw": "solved_steps",
    "baoding_p1_ccw": "solved_steps",
    "baoding_p2_overlap": "solved_steps",
    "baoding_p2": "solved_steps",
    "elbow_pose": "solved_steps",
    "relocate": "solved",
    "kinesis": "solved_steps",
}


def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def _collect_seed_results(method_dir):
    """Return the sorted per-seed *_results.json files under a final_benchmarks method dir."""
    pattern = os.path.join(
        ROOT_DIR, "data/final_benchmarks", method_dir, "*", "*_results.json"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results files found for method dir: {method_dir}")
    return files


def create_radar_plot():
    # Load results from data/final_benchmarks - one directory per method, with
    # one *_results.json per seed.
    filepath_dict = {
        "PPO": _collect_seed_results("ppo_t_sv"),
        "BC": _collect_seed_results("bc"),
        "Arnold (ours)": _collect_seed_results("arnold"),
    }

    color_dict = {
        "PPO": "#C44E52",
        "BC": "#CCB974",
        "Arnold (ours)": "#4C72B0",
    }

    expert_dir = os.path.join(ROOT_DIR, "data/final_benchmarks/expert_policies")

    # Load all results - now aggregating multiple files per method
    results_dict = {}
    for key, filepaths in filepath_dict.items():
        results_dict[key] = []
        for filepath in filepaths:
            results_dict[key].append(load_results(filepath))

    # Load expert results
    expert_results = {}
    for filename in os.listdir(expert_dir):
        if filename.endswith("_results.json"):
            results = load_results(os.path.join(expert_dir, filename))
            expert_results.update(results)

    # Get common tasks - using the first file of the first method
    tasks = sorted(list(results_dict[list(results_dict.keys())[0]][0].keys()))

    # Prepare data with mean and SEM
    scores_dict = {}
    scores_sem_dict = {}  # Store SEM values
    expert_scores = []

    for task in tasks:
        metric = TASK_METRIC_MAP[task]
        expert_score = expert_results[task][f"avg_{metric}"]
        expert_scores.append(100)  # Expert is always 100%

        # Calculate relative performance
        for key, results_list in results_dict.items():
            if key not in scores_dict:
                scores_dict[key] = []
                scores_sem_dict[key] = []

            # Collect scores from all files for this method and task
            all_scores = []
            for results in results_list:
                task_score = results.get(task)
                if task_score is not None:
                    all_scores.append(
                        (task_score[f"avg_{metric}"] / expert_score) * 100
                    )

            if all_scores:
                # Calculate mean and SEM
                mean_score = np.mean(all_scores)
                sem = (
                    np.std(all_scores, ddof=1) / np.sqrt(len(all_scores))
                    if len(all_scores) > 1
                    else 0
                )

                scores_dict[key].append(mean_score)
                scores_sem_dict[key].append(sem)
            else:
                scores_dict[key].append(0)
                scores_sem_dict[key].append(0)

    # Set up the angles for the radar plot
    angles = [n / float(len(tasks)) * 2 * np.pi for n in range(len(tasks))]
    angles += angles[:1]  # Complete the circle

    # Add the first value again to complete the circle
    for key in scores_dict:
        scores_dict[key] += scores_dict[key][:1]
        scores_sem_dict[key] += scores_sem_dict[key][:1]
    expert_scores += expert_scores[:1]

    # Create the plot with increased figure size for better spacing
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection="polar"))

    # Plot reference circles with extended range
    for percent in [50, 100]:
        circle = [percent] * (len(angles))
        ax.plot(angles, circle, color="grey", linestyle="--", alpha=1, linewidth=0.5)
        # Add percentage labels only on the right side
        ax.text(
            0.7,
            percent,
            f"{percent:.0f}%",
            ha="left",
            va="center",
            color="gray",
            alpha=1,
            fontsize=16,
        )

    # Plot expert performance markers for each task
    for i, angle in enumerate(angles[:-1]):  # Exclude the last duplicated angle
        # Plot vertical line from center to 100%
        ax.plot(
            [angle, angle],
            [0, 100],
            color="g",
            linestyle="-",
            alpha=0.2,
            linewidth=5,
        )
        # Add expert marker
        ax.plot(
            angle,
            100,
            "go",
            markersize=20,
            alpha=0.5,
            label="Single task experts" if i == 0 else "",
        )

    # Plot multi-task performance with shaded error bands
    for key, scores in scores_dict.items():
        sem_values = scores_sem_dict[key]
        ax.plot(
            angles,
            scores,
            linestyle="-",
            linewidth=5,
            label=f"{key}",
            color=color_dict[key],
        )

        # Add shaded error band (±SEM)
        upper_bound = np.array(scores) + np.array(sem_values)
        lower_bound = np.array(scores) - np.array(sem_values)
        ax.fill_between(
            angles, lower_bound, upper_bound, color=color_dict[key], alpha=0.2
        )

        # Print mean ± SEM across all tasks
        task_scores = scores[:-1]  # Exclude the duplicated first element
        print(
            f"{key}: {np.mean(task_scores):.2f} ± {(np.std(task_scores) / np.sqrt(len(task_scores))):.2f}"
        )

    # ------------------------------------------------------------------
    # Overall comparison: two-sided Wilcoxon signed-rank test across the N tasks,
    # Holm-Bonferroni-corrected over the 3 comparisons (BC, PPO, Expert). Each
    # task contributes its across-seed mean relative score; the expert is 100 by
    # construction, so the Arnold-vs-Expert test reduces to a signed-rank test of
    # the per-task relative scores against 100.
    # ------------------------------------------------------------------
    print("\nOverall comparison (Wilcoxon signed-rank across tasks, Holm):")
    overall_comparisons = ("BC", "PPO", "Expert")
    arnold_task_means = np.array(scores_dict["Arnold (ours)"][:-1])
    expert_task_means = np.full(len(arnold_task_means), 100.0)
    overall_raw_p = []
    for baseline in overall_comparisons:
        other = (
            expert_task_means
            if baseline == "Expert"
            else np.array(scores_dict[baseline][:-1])
        )
        overall_raw_p.append(
            stats.wilcoxon(arnold_task_means, other, alternative="two-sided").pvalue  # type: ignore[attr-defined]
        )
    for baseline, p in zip(overall_comparisons, holm_bonferroni(overall_raw_p)):
        print(f"  Arnold vs {baseline} [Holm, two-sided]: p={p:.4f}")

    # ------------------------------------------------------------------
    # Overall comparison (independent-samples view): Mann-Whitney U on the
    # per-seed mean relative scores. For each method, average the relative score
    # across tasks within each seed to get one value per seed (n=5); Arnold's
    # seeds are independent of each baseline's seeds, so an unpaired test applies.
    # Two-sided, Holm-corrected across the 2 comparisons (BC, PPO).
    #
    # Expert is a single constant per task with no per-seed distribution, so
    # Mann-Whitney U does not apply there (see the Wilcoxon result above). Note
    # that with n=5 vs n=5 the two-sided exact p-value floor is
    # 2 / C(10, 5) ~= 0.008, reached under complete separation.
    # ------------------------------------------------------------------
    print("\nOverall comparison (Mann-Whitney U on per-seed means, Holm):")
    seed_means = {}
    for key, results_list in results_dict.items():
        means = []
        for results in results_list:  # one entry per seed
            rel_scores = []
            for task in tasks:
                metric = TASK_METRIC_MAP[task]
                expert_score = expert_results[task][f"avg_{metric}"]
                task_score = results.get(task)
                if task_score is not None:
                    rel_scores.append((task_score[f"avg_{metric}"] / expert_score) * 100)
            means.append(np.mean(rel_scores))
        seed_means[key] = np.array(means)

    arnold_seed_means = seed_means["Arnold (ours)"]
    mwu_comparisons = ("BC", "PPO")
    mwu_raw_p = [
        stats.mannwhitneyu(
            arnold_seed_means,
            seed_means[baseline],
            alternative="two-sided",
            method="exact",
        ).pvalue
        for baseline in mwu_comparisons
    ]
    for baseline, p in zip(mwu_comparisons, holm_bonferroni(mwu_raw_p)):
        print(f"  Arnold vs {baseline} [Holm, two-sided]: p={p:.4f}")

    # ------------------------------------------------------------------
    # Per-task comparison: is Arnold better than the single-task expert?
    # For each task, form the per-seed improvement over the expert in relative
    # percentage points,
    #     improvement(k) = (Arnold_metric(k) / Expert_metric) * 100 - 100,
    # and run a ONE-SIDED one-sample t-test of H1: mean improvement > 0. The raw
    # per-task p-values are Holm-Bonferroni-corrected across the N tasks.
    #
    # The expert has a single value per task, so this is exactly a paired t-test
    # on Arnold(n, k) - Expert(n). Rescaling each task's differences by the
    # positive constant 100 / Expert(n) leaves the t-statistic (and hence the
    # p-value) unchanged, so raw-unit and relative improvements agree on p.
    # ------------------------------------------------------------------
    arnold_results = results_dict["Arnold (ours)"]
    task_stats = []  # (task, mean_improvement, std_improvement, raw_p)
    for task in tasks:
        metric = TASK_METRIC_MAP[task]
        expert_score = expert_results[task][f"avg_{metric}"]
        improvements = []
        for results in arnold_results:  # one entry per seed
            task_score = results.get(task)
            if task_score is not None:
                improvements.append(
                    (task_score[f"avg_{metric}"] / expert_score) * 100 - 100
                )
        improvements = np.array(improvements)
        mean_imp = float(np.mean(improvements))
        std_imp = float(np.std(improvements, ddof=1)) if len(improvements) > 1 else 0.0
        if std_imp > 0:
            raw_p = float(
                stats.ttest_1samp(improvements, 0.0, alternative="greater").pvalue  # type: ignore[attr-defined]
            )
        else:
            # Zero variance makes the t-test undefined; treat a strictly positive
            # constant improvement as maximal evidence and a non-positive one as none.
            raw_p = 0.0 if mean_imp > 0 else 1.0
        task_stats.append((task, mean_imp, std_imp, raw_p))

    adj_p_all = holm_bonferroni([s[3] for s in task_stats])

    print(
        "\nPer-task comparison (Arnold vs Expert, one-sided one-sample t on "
        "improvement, Holm):"
    )
    for (task, mean_imp, std_imp, raw_p), adj_p in zip(task_stats, adj_p_all):
        flag = " *" if adj_p < 0.05 else ""
        print(
            f"  {task:<20s}: improvement {mean_imp:+6.2f} ± {std_imp:5.2f}  "
            f"raw p={raw_p:.4f}  Holm p={adj_p:.4f}{flag}"
        )

    # LaTeX table: per-task mean improvement over the expert (± std across seeds)
    # and the Holm-corrected one-sided p-value.
    task_to_stats = {s[0]: s for s in task_stats}
    task_to_adj = {s[0]: p for s, p in zip(task_stats, adj_p_all)}
    ordered_tasks = [t for t in TASK_NAME_MAPPING if t in task_to_stats]
    tex_lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Task & Improvement (\\%) & Holm $p$ \\\\",
        "\\midrule",
    ]
    for t in ordered_tasks:
        _, mean_imp, std_imp, _ = task_to_stats[t]
        name = " ".join(TASK_NAME_MAPPING[t].split())
        p_adj = task_to_adj[t]
        # Star significant (Holm p < 0.05) values, e.g. $0.0079^{*}$. Floor the
        # displayed value so sub-1e-4 p-values read "<0.0001" instead of "0.0000".
        p_disp = f"{p_adj:.4f}" if p_adj >= 1e-4 else "<0.0001"
        p_str = f"${p_disp}^{{*}}$" if p_adj < 0.05 else f"${p_disp}$"
        tex_lines.append(
            f"{name} & ${mean_imp:+.2f} \\pm {std_imp:.2f}$ & {p_str} \\\\"
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}"]
    tex_table = "\n".join(tex_lines)
    tex_path = os.path.join(ROOT_DIR, "data/figures/arnold_vs_expert_improvement.tex")
    with open(tex_path, "w") as f:
        f.write(tex_table + "\n")
    print("\nLaTeX table (Arnold vs Expert, improvement ± std and Holm p-values):")
    print(tex_table)
    print(f"Saved to {tex_path}")

    # Add gridlines
    ax.grid(False)  # Remove the default grid
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color("lightgray")

    # Set the labels with mapped task names and improved positioning
    ax.set_xticks(angles[:-1])
    mapped_tasks = [TASK_NAME_MAPPING.get(task, task) for task in tasks]
    # Adjust label positions with more margin and better alignment
    ax.set_xticklabels(
        mapped_tasks,
        y=-0.1,  # More margin from the plot
        fontsize=32,
        horizontalalignment="center",
        verticalalignment="center",
    )

    # Configure axis limits to show values above 100%
    ax.set_ylim(0, 130)
    ax.set_yticklabels([])

    # Adjust legend position and style
    # legend = ax.legend(
    #     loc="upper center", bbox_to_anchor=(0.5, 1.3), fontsize=28, framealpha=0.8, ncol=2
    # )

    # Save the plot
    plt.tight_layout()
    plt.savefig(
        os.path.join(ROOT_DIR, "data/figures/radar_plot_ppo_bc_arnold.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        os.path.join(ROOT_DIR, "data/figures/radar_plot_ppo_bc_arnold.svg"),
        bbox_inches="tight",
        dpi=300,
    )
    print("Radar plot saved successfully to data/figures/radar_plot_ppo_bc_arnold.png and .svg")
    plt.close()


if __name__ == "__main__":
    create_radar_plot()
