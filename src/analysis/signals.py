"""Lossless episode storage shared by collection and export."""
import json
from pathlib import Path

import h5py
import numpy as np


TIMING = {
    "actions": "before_step",
    "action_means": "before_step",
    "joint_positions": "before_step",
    "muscle_controls": "after_step",
    "muscle_activations": "after_step",
    "rewards": "after_step",
    "solved": "after_step",
}


def initialize_file(file, metadata, axes):
    file.attrs["schema_version"] = 1
    file.attrs.update(metadata)
    for name, values in axes.items():
        file.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype()))


def write_episode(file, episode_id, signals, timing=None):
    group = file.create_group(f"episode_{episode_id}")
    length = len(signals["rewards"])
    group.attrs.update(episode_length=length,
                       total_reward=np.sum(signals["rewards"]),
                       solved_steps=np.sum(signals["solved"]))
    for name, values in signals.items():
        dataset = group.create_dataset(name, data=values, compression="gzip", shuffle=True)
        dataset.attrs["timing"] = (timing or {}).get(name, TIMING.get(name, "before_step"))
        if name == "joint_positions":
            dataset.attrs["units"] = "radian"
    return group


def read_recording(file, selected, joint_count):
    length = int(file.attrs.get("episode_length", len(file["rewards"])))
    signals = {}
    for name in selected:
        if name == "joint_positions":
            observations = file["observations"]
            if isinstance(observations, h5py.Group):
                values = observations["obs"][:length, :joint_count, -1]
            else:
                values = observations[:length, :joint_count]
        else:
            key = "activations" if name == "muscle_controls" else name
            values = file[key][:length]
            if name in ("actions", "action_means"):
                values = values.reshape(len(values), -1)
        signals[name] = values
    return signals


def selection_paths(data_dir, analysis, selections_path="data/reproduction/signals.json"):
    selections = json.loads(Path(selections_path).read_text())
    selection = selections[analysis]
    for policy in selection["policies"]:
        for task in selection.get("tasks", selections["tasks"]):
            collection = selection.get("overrides", {}).get(policy, {}).get(task, policy)
            yield policy, task, Path(data_dir) / collection / f"{task}.h5"


def iter_episodes(path, signals):
    with h5py.File(path, "r") as file:
        names = sorted((key for key in file if key.startswith("episode_")),
                       key=lambda key: int(key.split("_")[-1]))
        for name in names:
            group = file[name]
            length = int(group.attrs["episode_length"])
            yield int(name.split("_")[-1]), {key: group[key][:length] for key in signals}
