"""Control subspaces and episode-based performance."""
from contextlib import contextmanager

import numpy as np
from analysis.signals import iter_episodes


def load_signals(path, signal, successful=None):
    if successful is not None and successful <= 0:
        raise ValueError("successful episode count must be positive")
    values = []
    selected = [signal, "solved"] if successful is not None else [signal]
    for _, episode in iter_episodes(path, selected):
        if successful is not None and not np.any(episode["solved"]):
            continue
        length = len(episode[signal])
        if length:
            values.append(episode[signal].reshape(length, -1))
        if successful is not None and len(values) == successful:
            break
    if successful is not None and len(values) < successful:
        raise ValueError(f"{path}: requested {successful} successful episodes, found {len(values)}")
    if not values:
        raise ValueError(f"No nonempty episodes in {path}")
    return np.vstack(values), len(values)


def principal_components(covariance):
    eigenvalues, vectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], vectors[:, order].T


def compare_subspaces(covariance, components, other_components, dimension):
    own = components[:dimension]
    other = other_components[:dimension]
    own_variance = np.trace(own @ covariance @ own.T)
    projected_variance = np.trace(other @ covariance @ other.T)
    pvd = 1 - projected_variance / own_variance if own_variance else 0.0
    cosines = np.clip(np.linalg.svd(own @ other.T, compute_uv=False), 0, 1)
    pad = np.degrees(np.arccos(cosines)).mean()
    return float(pvd), float(pad)


def project_actions(actions, mean, components):
    return (actions - mean) @ components.T @ components + mean


@contextmanager
def nmf_controls(env, nmf):
    base = env.unwrapped
    base = getattr(base, "gym_env", base).unwrapped
    robot = base.robot
    original = robot.process_actuator

    def process(*args, **kwargs):
        controls = original(*args, **kwargs)
        return nmf.inverse_transform(nmf.transform(controls.astype(float).reshape(1, -1)))[0]

    robot.process_actuator = process
    try:
        yield
    finally:
        robot.process_actuator = original


def performance_summary(episodes, horizon):
    fractions = np.asarray([row["solved_steps"] / horizon for row in episodes])
    return {"solved_fraction": fractions.mean(),
            "solved_fraction_sem": fractions.std() / np.sqrt(len(fractions)),
            "n_episodes": len(episodes)}
