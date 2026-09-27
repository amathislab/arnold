import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis.subspaces import compare_subspaces, principal_components, project_actions, nmf_controls, performance_summary


class SubspacesTest(unittest.TestCase):
    def test_directed_variance_and_orthogonal_angles(self):
        covariance = np.diag([9., 1.])
        _, components = principal_components(covariance)
        pvd, pad = compare_subspaces(covariance, components, components[::-1], 1)
        self.assertAlmostEqual(pvd, 8/9)
        self.assertAlmostEqual(pad, 90)
        pvd, pad = compare_subspaces(covariance, components, components[::-1], 2)
        self.assertAlmostEqual(pvd, 0)
        self.assertAlmostEqual(pad, 0)

    def test_full_rank_projection_preserves_nonzero_mean(self):
        rng = np.random.RandomState(4)
        values = rng.normal(size=(40, 5)) + 8
        _, components = principal_components(np.cov(values.T))
        np.testing.assert_allclose(project_actions(values, values.mean(0), components), values, atol=1e-12)

    def test_nmf_hook_processes_feasible_controls_and_restores_method(self):
        class Robot:
            def process_actuator(self, controls):
                return np.clip(controls, 0, 1)
        class Env:
            unwrapped = property(lambda self: self)
            robot = Robot()
        class Factorization:
            def transform(self, values):
                np.testing.assert_array_equal(values, [[0, 1]])
                return values
            def inverse_transform(self, values):
                return values / 2
        env = Env()
        with nmf_controls(env, Factorization()):
            np.testing.assert_array_equal(env.robot.process_actuator(np.array([-1, 2])), [0, .5])
        np.testing.assert_array_equal(env.robot.process_actuator(np.array([-1, 2])), [0, 1])

    def test_performance_uses_fixed_horizon(self):
        result = performance_summary([{"solved_steps": 20, "steps": 20}, {"solved_steps": 0, "steps": 200}], 200)
        self.assertAlmostEqual(result["solved_fraction"], .05)
        self.assertAlmostEqual(result["solved_fraction_sem"], .05 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
