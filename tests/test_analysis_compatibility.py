"""Regression checks for adapting PR analyses to the simplified repository."""
import json
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.signals import iter_episodes, initialize_file, write_episode
from analysis.subspaces import load_signals
from analysis.rollouts import get_episode_horizon
from analysis.export_signals import export_recordings
from analysis.import_human_emg import import_profiles


class CompatibilityTests(unittest.TestCase):
    def test_legacy_and_compact_recordings_produce_identical_pca_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            axes = {"joint_names": [str(i) for i in range(23)]}
            for episode in [2, 10]:
                with h5py.File(root / f"pen_episode_{episode}.h5", "w") as file:
                    file.attrs["episode_length"] = 5
                    file["rewards"] = np.ones(8)
                    file["solved"] = np.ones(8)
                    file["action_means"] = np.arange(8 * 39).reshape(8, 1, 39)
                    file["observations/obs"] = np.arange(8 * 23 * 6).reshape(8, 23, 6)
            episodes = list(iter_episodes(root / "pen.h5", ["joint_positions"]))
            self.assertEqual([n for n, _ in episodes], [2, 10])
            self.assertEqual(episodes[0][1]["joint_positions"].shape, (5, 23))
            legacy, count = load_signals(root / "pen.h5", "action_means", successful=2)
            export_recordings(root, root / "compact.h5", "pen", "legacy", [2, 10],
                              ["action_means", "joint_positions"], axes, 200, .02)
            compact, compact_count = load_signals(root / "compact.h5", "action_means", successful=2)
            np.testing.assert_array_equal(legacy, compact)
            self.assertEqual((count, compact_count), (2, 2))
            self.assertEqual(compact.shape, (10, 39))

    def test_success_selection_does_not_cross_episode_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pen.h5"
            with h5py.File(path, "w") as file:
                initialize_file(file, {}, {})
                for episode, solved in enumerate([False, True, False, True]):
                    write_episode(file, episode, {"actions": np.full((3, 2), episode),
                                  "rewards": np.ones(3), "solved": np.full(3, solved)})
            values, count = load_signals(path, "actions", successful=2)
            np.testing.assert_array_equal(values[:, 0], [1, 1, 1, 3, 3, 3])
            self.assertEqual(count, 2)

    def test_horizon_follows_gymnasium_and_shimmy_wrappers(self):
        from types import SimpleNamespace
        base = SimpleNamespace(spec=SimpleNamespace(max_episode_steps=200))
        env = SimpleNamespace(env=SimpleNamespace(gym_env=base))
        self.assertEqual(get_episode_horizon(env), 200)
        base.env = env
        base.spec = None
        with self.assertRaises(ValueError):
            get_episode_horizon(env)

    def test_human_import_preserves_subject_and_muscle_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = ROOT / "data/analysis/reproduction/emg_muscles.json"
            for subject in range(4, 13):
                np.save(root / f"EMG_subject_{subject:02d}_walk_45_avg.npy",
                        np.tile(np.arange(10) + subject * 100, (20, 1)))
            import_profiles(root, root / "human.npz", mapping, local_only=True)
            with np.load(root / "human.npz") as data:
                np.testing.assert_array_equal(data["subjects"], np.arange(4, 13))
                self.assertEqual(data["profiles"].shape, (9, 20, 9))
                np.testing.assert_array_equal(data["profiles"][0, 0], np.arange(1, 10) + 400)
                self.assertEqual(list(data["muscles"]), list(json.loads(mapping.read_text())["emg"]))

    def test_human_import_prefers_published_profiles(self):
        mapping = ROOT / "data/analysis/reproduction/emg_muscles.json"
        labels = ["Time", *json.loads(mapping.read_text())["emg"]]

        def download(url, timeout):
            buffer = BytesIO()
            np.save(buffer, np.array(labels) if url.endswith("EMG_labels.npy") else np.ones((100, 10)))
            buffer.seek(0)
            return buffer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in range(4, 13):
                np.save(root / f"EMG_subject_{subject:02d}_walk_45_avg.npy", np.zeros((100, 10)))
            with patch("analysis.import_human_emg.urlopen", side_effect=download) as request:
                import_profiles(root, root / "human.npz", mapping)
            self.assertEqual(request.call_count, 10)
            with np.load(root / "human.npz") as data:
                np.testing.assert_array_equal(data["profiles"], np.ones((9, 100, 9)))

    def test_human_import_network_failure_uses_complete_local_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in range(4, 13):
                np.save(root / f"EMG_subject_{subject:02d}_walk_45_avg.npy", np.full((100, 10), subject))
            with patch("analysis.import_human_emg.urlopen", side_effect=URLError("offline")):
                with self.assertWarnsRegex(UserWarning, "using local profiles"):
                    import_profiles(root, root / "human.npz")
            with np.load(root / "human.npz") as data:
                np.testing.assert_array_equal(data["profiles"][:, 0, 0], np.arange(4, 13))
            (root / "EMG_subject_12_walk_45_avg.npy").unlink()
            with patch("analysis.import_human_emg.urlopen", side_effect=URLError("offline")):
                with self.assertRaisesRegex(RuntimeError, "no complete local cohort"):
                    import_profiles(root, root / "missing.npz")
            self.assertFalse((root / "missing.npz").exists())

    def test_human_import_rejects_wrong_published_muscle_order(self):
        buffer = BytesIO()
        np.save(buffer, np.array(["wrong labels"]))
        payload = buffer.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            with patch("analysis.import_human_emg.urlopen", side_effect=lambda *a, **kw: BytesIO(payload)):
                with self.assertRaisesRegex(ValueError, "muscle order"):
                    import_profiles(directory, Path(directory) / "human.npz")


if __name__ == "__main__":
    unittest.main()
