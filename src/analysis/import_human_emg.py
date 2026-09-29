"""Import Mathis Lab human EMG profiles, with a local/offline fallback."""
import argparse
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen
import warnings

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_URL = "https://huggingface.co/datasets/amathislab/kinesis-assets/resolve/main/emg_assets/human_emg"
DEFAULT_INPUT = REPO_ROOT / "data/analysis/emg/human_profiles"
DEFAULT_OUTPUT = REPO_ROOT / "data/analysis/emg/human.npz"
DEFAULT_MUSCLE_MAP = REPO_ROOT / "data/analysis/reproduction/emg_muscles.json"


def import_profiles(input_dir=DEFAULT_INPUT, output=DEFAULT_OUTPUT,
                    muscle_map=DEFAULT_MUSCLE_MAP, *, local_only=False):
    """Prefer published assets; fall back to a complete local cohort on network errors.

    Set local_only=True to import custom/offline files without accessing the network.
    Invalid remote data raises an error rather than silently substituting local data.
    """
    subjects = np.arange(4, 13)
    profiles = []
    muscles = list(json.loads(Path(muscle_map).read_text())["emg"])
    filenames = [f"EMG_subject_{subject:02d}_walk_45_avg.npy" for subject in subjects]
    arrays = None
    if not local_only:
        try:
            payloads = {}
            for name in ["EMG_labels.npy", *filenames]:
                with urlopen(f"{ASSET_URL}/{name}", timeout=30) as response:
                    payloads[name] = response.read()
        except OSError as error:
            if not all((Path(input_dir) / name).is_file() for name in filenames):
                raise RuntimeError(
                    f"Could not download Mathis Lab EMG assets and no complete local "
                    f"cohort exists at {input_dir}. Supply --input_dir or retry online."
                ) from error
            warnings.warn(f"Mathis Lab download failed ({error}); using local profiles at {input_dir}")
        else:
            labels = np.load(BytesIO(payloads["EMG_labels.npy"]), allow_pickle=False)
            if labels.tolist() != ["Time", *muscles]:
                raise ValueError("Mathis Lab EMG labels do not match the configured muscle order")
            arrays = [np.load(BytesIO(payloads[name]), allow_pickle=False) for name in filenames]
    if arrays is None:
        arrays = [np.load(Path(input_dir) / name, allow_pickle=False) for name in filenames]
    for name, values in zip(filenames, arrays):
        if values.ndim != 2 or values.shape[1] != len(muscles) + 1:
            raise ValueError(f"{name}: expected time plus {len(muscles)} muscle columns")
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: nonfinite profile values")
        profiles.append(values[:, 1:])
    profiles = np.stack(profiles)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, profiles=profiles, subjects=subjects, muscles=muscles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT,
                        help="Local fallback directory (also used with --local-only)")
    parser.add_argument("--local-only", action="store_true", help="Skip downloading; use only --input_dir")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--muscle_map", type=Path, default=DEFAULT_MUSCLE_MAP)
    args = parser.parse_args()
    import_profiles(args.input_dir, args.output, args.muscle_map, local_only=args.local_only)


if __name__ == "__main__":
    main()
