import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plotting"))
from analysis.performance import mean_sem, holm, rank_biserial
from plot_csi_analysis import summarize_runs


class PerformanceTest(unittest.TestCase):
    def test_seed_sem_uses_sample_standard_deviation(self):
        mean, sem = mean_sem([90, 100, 110])
        self.assertEqual(mean, 100)
        self.assertAlmostEqual(sem, 10 / np.sqrt(3))

    def test_holm_preserves_order_and_monotonicity(self):
        np.testing.assert_allclose(holm([.04, .01, .03]), [.06, .03, .06])

    def test_rank_biserial_handles_ties(self):
        self.assertAlmostEqual(rank_biserial([1, -1, 2, 0]), .5)

    def test_missing_run_is_distinct_from_measured_zero(self):
        missing = summarize_runs([])
        self.assertTrue(np.isnan(missing['mean']))
        self.assertEqual(missing['n_runs'], 0)
        measured = summarize_runs([0, .6])
        self.assertAlmostEqual(measured['mean'], .3)
        self.assertEqual(measured['n_runs'], 2)


if __name__ == "__main__":
    unittest.main()
