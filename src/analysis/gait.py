"""Heel-strike segmentation and gait-normalized muscle profiles."""
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt


def lowpass(values, fs=30.0, cutoff=5.0, order=5):
    b, a = butter(order, cutoff / (0.5 * fs), btype="low")
    return filtfilt(b, a, values, axis=0)


def strikes(contact, fs, threshold=300.0):
    above = lowpass(np.maximum(contact, 0).astype(float), fs) > threshold
    return np.flatnonzero(~above[:-1] & above[1:]) + 1


def resample(values, length=200, axis=0):
    return interp1d(np.linspace(0, 1, values.shape[axis]), values, axis=axis)(
        np.linspace(0, 1, length))


def mean_gait(episodes, side="left", fs=30.0):
    """Alternate heel strikes; central 80% durations; forward speed >= 0.2 m/s."""
    origin, opposite = (2, 0) if side == "left" else (0, 2)
    cycles, detected = [], 0
    for episode in episodes:
        contacts = episode["foot_contacts"]
        if len(contacts) < 19:
            continue
        starts, others = strikes(contacts[:, origin], fs), strikes(contacts[:, opposite], fs)
        pairs = [(start, end) for start, end in zip(starts[:-1], starts[1:])
                 if np.any((others > start) & (others < end))]
        detected += len(pairs)
        if not pairs:
            continue
        low, high = np.percentile([end - start for start, end in pairs], [10, 90])
        forward = lowpass(episode["root_vel"].astype(float), fs)[:, 0]
        for start, end in pairs:
            if low <= end - start <= high and forward[start:end].mean() >= 0.2:
                cycles.append(resample(episode["activations"][start:end]))
    if not cycles:
        raise ValueError("No gait cycles meet the selection criteria")
    return np.mean(np.stack(cycles), axis=0).T, {"detected": detected, "retained": len(cycles)}
