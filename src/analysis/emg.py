"""Human-to-policy correlations and the leave-one-out human reference."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis.gait import resample


def correlations(human, simulated):
    """Inputs: subject x time x muscle and muscle x time arrays."""
    simulated = resample(simulated.T, human.shape[1])
    return np.array([[np.corrcoef(subject[:, muscle], simulated[:, muscle])[0, 1]
                      for muscle in range(human.shape[2])] for subject in human])


def leave_one_out(human):
    return np.array([[np.corrcoef(subject[:, muscle],
                                 np.delete(human, index, axis=0).mean(axis=0)[:, muscle])[0, 1]
                      for muscle in range(human.shape[2])]
                     for index, subject in enumerate(human)])


def policy_statistics(subject_means):
    rows = []
    for first, second in combinations(subject_means, 2):
        differences = np.arctanh(subject_means[first]) - np.arctanh(subject_means[second])
        statistic, pvalue = (0.0, 1.0) if np.all(differences == 0) else wilcoxon(
            differences, alternative="two-sided")
        rows.append({"a": first, "b": second, "n_subjects": len(differences),
                     "W": float(statistic), "p_raw": float(pvalue),
                     "p_bonferroni": min(1.0, 3 * float(pvalue))})
    return pd.DataFrame(rows)
