"""Checks for episode boundaries, covariance PCA and the paired task unit."""
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.kinematics import dimensionality, pooled_kinematics
from analysis.signals import iter_episodes
from analysis.smoothness import METRICS, paired_statistics


class HandAnalysisTests(unittest.TestCase):
    def test_velocity_uses_executed_steps_within_each_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "signals.h5"
            with h5py.File(path, "w") as file:
                for episode, offset in ((2, 0.0), (10, 100.0)):
                    group = file.create_group(f"episode_{episode}")
                    group.attrs["episode_length"] = 3
                    group.create_dataset("joint_positions", data=np.array([
                        [offset, offset], [offset + 1, offset + 2],
                        [offset + 2, offset + 4], [9999, 9999]]))
            episodes = list(iter_episodes(path, ["joint_positions"]))
        self.assertEqual([episode for episode, _ in episodes], [2, 10])
        trajectories = [signals["joint_positions"] for _, signals in episodes]
        self.assertEqual(pooled_kinematics(trajectories, "position").shape, (6, 2))
        np.testing.assert_array_equal(pooled_kinematics(trajectories, "velocity"),
                                      np.tile([1.0, 2.0], (4, 1)))

    def test_pca_centers_raw_angles_without_standardizing_joints(self):
        time = np.arange(100) * 2 * np.pi / 100
        samples = np.column_stack((100 * np.sin(time), np.cos(time)))
        counts, curve = dimensionality(samples)
        self.assertEqual(counts, [1, 1])
        shifted_counts, shifted_curve = dimensionality(samples + [30, -80])
        self.assertEqual(shifted_counts, counts)
        np.testing.assert_allclose(shifted_curve, curve, atol=1e-12)

    def test_comparisons_pair_tasks_independent_of_episode_counts(self):
        policies, tasks = ["expert", "single_task", "obc"], list(range(11))
        rows = []
        for task in tasks:
            for index, policy in enumerate(policies):
                for episode in range(1 if task else 100):
                    rows.append({"policy": policy, "task": task, "episode": episode,
                                 **{metric: float(task + index) for metric in METRICS}})
        frame = pd.DataFrame(rows)
        means = frame.groupby(["policy", "task"])[list(METRICS)].mean()
        comparisons = paired_statistics(means, policies, tasks)
        self.assertEqual(len(comparisons), 12)
        np.testing.assert_array_equal(comparisons.n_tasks, 11)
        np.testing.assert_array_equal(comparisons.W, 0)
        np.testing.assert_allclose(comparisons.p_raw, 2 / 2 ** 11)
        np.testing.assert_allclose(comparisons.p_holm, 3 * 2 / 2 ** 11)
        with self.assertRaises(ValueError):
            paired_statistics(means.drop(("expert", 0)), policies, tasks)


if __name__ == "__main__":
    unittest.main()
