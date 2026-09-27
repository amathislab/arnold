"""Benchmark summaries with seeds as the unit of replication."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]


def task_names():
    return json.loads((REPO_ROOT / "data/reproduction/tasks.json").read_text())


def load_benchmarks(method, root=REPO_ROOT):
    paths = json.loads((REPO_ROOT / "data/reproduction/policies.json").read_text())[method]
    return [json.loads((Path(root) / path).read_text()) for path in paths]


def expert_means(root=REPO_ROOT):
    return {task: json.loads((Path(root) / "data/final_benchmarks/expert_policies" /
                             f"expert_{task}_results.json").read_text())[task]["avg_solved_step_frac"]
            for task in task_names()}


def relative_matrix(method, tasks, root=REPO_ROOT):
    experts = expert_means(root)
    return np.asarray([[result[task]["avg_solved_step_frac"] / experts[task] * 100
                        for task in tasks] for result in load_benchmarks(method, root)])


def mean_sem(values):
    values = np.asarray(values)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")


def holm(pvalues):
    pvalues = np.asarray(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min((len(pvalues) - rank) * pvalues[index], 1.))
        adjusted[index] = running
    return adjusted


def rank_biserial(differences):
    values = np.asarray(differences)
    values = values[values != 0]
    ranks = rankdata(np.abs(values))
    return float(np.sum(ranks * np.sign(values)) / ranks.sum()) if len(values) else 0.
