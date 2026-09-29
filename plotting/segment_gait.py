"""Build compact EMG and factor-analysis profiles from gait recordings."""
import _srcpath  # noqa: F401
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from analysis.gait import mean_gait


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=Path("data/analysis/emg/rollouts"))
    parser.add_argument("--output", type=Path, default=Path("data/analysis/emg/simulation.npz"))
    parser.add_argument("--muscle_map", type=Path, default=Path("data/analysis/reproduction/emg_muscles.json"))
    args = parser.parse_args()
    mapping = json.loads(args.muscle_map.read_text())
    output, statistics = {}, {}
    for policy in ("arnold", "obc", "kinesis"):
        with h5py.File(args.rollouts / f"{policy}.h5", "r") as file:
            names = list(file["muscle_names"].asstr()[:])
            groups = sorted((key for key in file if key.startswith("episode_")),
                            key=lambda key: int(key.split("_")[-1]))
            episodes = [{key: file[group][key][:int(file[group].attrs["episode_length"])]
                         for key in ("activations", "foot_contacts", "root_vel")} for group in groups]
            fs = float(file.attrs["fs"])
        mean, counts = mean_gait(episodes, side="left", fs=fs)
        output[policy + "_left"] = mean[[names.index(name) for name in mapping["emg"].values()]]
        statistics[policy + "_left"] = counts
        if policy == "arnold":
            mean, counts = mean_gait(episodes, side="right", fs=fs)
            output["arnold_right"] = np.stack([mean[[names.index(name) for name in group]].sum(axis=0)
                                              for group in mapping["factors"].values()])
            statistics["arnold_right"] = counts
    output["emg_muscles"] = np.array(list(mapping["emg"]))
    output["factor_muscles"] = np.array(list(mapping["factors"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    args.output.with_suffix(".json").write_text(json.dumps(statistics, indent=2) + "\n")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
