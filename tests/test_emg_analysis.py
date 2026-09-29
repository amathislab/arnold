"""Method checks for the human reference, gait boundaries and factor rotation."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.emg import leave_one_out
from analysis.gait import mean_gait
from analysis.gait_factors import varimax_rotation


class EmgAnalysisTests(unittest.TestCase):
    def test_reference_correlates_each_subject_with_the_other_subjects_mean(self):
        time = np.linspace(0, 2 * np.pi, 100)
        human = np.array([np.column_stack((np.sin(time + shift), np.cos(time + shift)))
                          for shift in (0.0, 0.3, 0.8)])
        actual = leave_one_out(human)
        expected = np.array([[np.corrcoef(human[index, :, muscle],
                          human[[other for other in range(3) if other != index], :, muscle].mean(axis=0))[0, 1]
                          for muscle in range(2)] for index in range(3)])
        np.testing.assert_allclose(actual, expected)
        pair_mean = np.mean([np.corrcoef(human[0, :, 0], human[other, :, 0])[0, 1] for other in [1, 2]])
        self.assertGreater(abs(actual[0, 0] - pair_mean), 0.01)

    def test_cycles_require_an_opposite_heel_strike_in_the_same_episode(self):
        time = np.arange(300)
        contacts = np.zeros((300, 4))
        contacts[:, 2] = 600 * ((time % 30) < 12)
        episode = {"foot_contacts": contacts, "root_vel": np.tile([1.0, 0, 0], (300, 1)),
                   "activations": np.column_stack((np.sin(time / 5), np.cos(time / 5)))}
        with self.assertRaises(ValueError):
            mean_gait([episode])
        contacts[:, 0] = 600 * (((time + 15) % 30) < 12)
        mean, counts = mean_gait([episode])
        self.assertEqual(mean.shape, (2, 200))
        self.assertGreater(counts["retained"], 0)

    def test_varimax_preserves_the_retained_subspace_reconstruction(self):
        random = np.random.RandomState(3)
        loadings, scores = random.randn(14, 4), random.randn(4, 200)
        rotated, rotation = varimax_rotation(loadings)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(rotated @ (rotation.T @ scores), loadings @ scores, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
