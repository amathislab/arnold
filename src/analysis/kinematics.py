"""Covariance PCA of pooled hand positions and within-episode velocities."""
import numpy as np
from sklearn.decomposition import PCA


THRESHOLDS = (0.85, 0.95)


def pooled_kinematics(trajectories, quantity):
    if quantity == "velocity":
        trajectories = [np.diff(values, axis=0) for values in trajectories]
    return np.concatenate(trajectories, axis=0)


def dimensionality(samples):
    """Center raw coordinates, fit PCA, and average the 85% and 95% counts."""
    pca = PCA(svd_solver="full").fit(samples)
    curve = np.cumsum(pca.explained_variance_ratio_)
    if not np.isfinite(curve).all():
        raise ValueError("PCA requires nonzero finite kinematic variance")
    counts = [int(np.searchsorted(curve, threshold) + 1) for threshold in THRESHOLDS]
    return counts, curve
