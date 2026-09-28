"""The four hand smoothness metrics and task-paired comparisons."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import wilcoxon


METRICS = ("mean_abs_diff", "sparc", "ldlj_kin", "nvp")


def sparc(velocity, fs=1.0, padlevel=4, fc=10.0, amp_th=0.05):
    """Spectral arc length of joint-velocity magnitude."""
    velocity = np.asarray(velocity, dtype=float)
    if velocity.size < 4 or np.allclose(velocity, 0):
        return np.nan
    size = int(2 ** (np.ceil(np.log2(velocity.size)) + padlevel))
    spectrum = np.abs(np.fft.rfft(velocity, n=size))
    spectrum /= spectrum.max()
    frequencies = np.fft.rfftfreq(size, d=1.0 / fs)
    band = frequencies <= fc
    spectrum, frequencies = spectrum[band], frequencies[band]
    above = np.flatnonzero(spectrum >= amp_th)
    if not len(above):
        return np.nan
    spectrum, frequencies = spectrum[:above[-1] + 1], frequencies[:above[-1] + 1]
    if len(spectrum) < 2:
        return np.nan
    df = np.diff(frequencies) / (frequencies[-1] - frequencies[0] + 1e-12)
    return float(-np.sum(np.sqrt(df ** 2 + np.diff(spectrum) ** 2)))


def compute_metrics(actions, joints):
    """Compute per-episode metrics at unit timestep (one simulation step)."""
    joints = np.asarray(joints, dtype=np.float64)
    if len(joints) < 4:
        return dict.fromkeys(METRICS, np.nan)
    velocity = np.diff(joints, axis=0)
    jerk = np.diff(joints, n=3, axis=0)
    duration = len(joints) - 1
    excursion = np.max(np.abs(joints - joints.mean(axis=0, keepdims=True)), axis=0)
    jerk_integral = np.sum(jerk ** 2, axis=0)
    moving = (excursion >= 1e-6) & (jerk_integral >= 1e-12)
    ldlj = -np.log(duration ** 5 / excursion[moving] ** 2 * jerk_integral[moving])
    peaks = []
    for speed in np.abs(velocity).T:
        if speed.max() >= 1e-6:
            peaks.append(len(find_peaks(speed, height=0.05 * speed.max())[0]))
    return {
        "mean_abs_diff": float(np.mean(np.abs(np.diff(actions, axis=0)))),
        "sparc": sparc(np.linalg.norm(velocity, axis=1)),
        "ldlj_kin": float(np.mean(ldlj)) if len(ldlj) else np.nan,
        "nvp": float(np.mean(peaks)) if peaks else np.nan,
    }


def holm_correct(pvalues):
    """Holm correction in the original comparison order."""
    corrected = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(np.argsort(pvalues)):
        running = min(1.0, max(running, (len(pvalues) - rank) * pvalues[index]))
        corrected[index] = running
    return corrected


def paired_statistics(per_task, policies, tasks):
    """Two-sided Wilcoxon over task means; Holm across three pairs per metric."""
    rows = []
    for metric in METRICS:
        values = per_task[metric].unstack("policy").reindex(index=tasks, columns=policies)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Incomplete task means for {metric}")
        comparisons = []
        for first, second in combinations(policies, 2):
            differences = values[first].to_numpy() - values[second].to_numpy()
            statistic, pvalue = (0.0, 1.0) if np.all(differences == 0) else wilcoxon(
                differences, zero_method="pratt", alternative="two-sided")
            comparisons.append({"metric": metric, "a": first, "b": second,
                                "n_tasks": len(tasks), "W": float(statistic),
                                "p_raw": float(pvalue)})
        for comparison, corrected in zip(comparisons, holm_correct([r["p_raw"] for r in comparisons])):
            rows.append({**comparison, "p_holm": float(corrected)})
    return pd.DataFrame(rows)
