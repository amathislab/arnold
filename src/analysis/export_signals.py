"""Export selected episode signals from HDF5 recordings."""
import argparse
import json
from pathlib import Path
import sys

import h5py

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.signals import initialize_file, read_recording, write_episode


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--policy_id", required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=None)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--signals", nargs="+", default=["actions", "action_means", "joint_positions"],
                        choices=["actions", "action_means", "joint_positions", "muscle_controls"])
    parser.add_argument("--axes", type=Path, help="Optional joint_names/muscle_names JSON; otherwise use recording metadata or the task model")
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--actions_clipped", action="store_true", default=None)
    parser.add_argument("--means_clipped", action="store_true", default=None)
    return parser.parse_args()


def export_recordings(input_dir, output, task, policy_id, episodes, selected, axes, horizon, dt, actions_clipped=None, means_clipped=None):
    selected = list(dict.fromkeys([*selected, "rewards", "solved"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "x") as destination:
        initialize_file(destination, {"policy_id": policy_id, "task": task,
                                      "horizon": horizon, "dt": dt}, axes)
        if actions_clipped is not None:
            destination.attrs["actions_clipped"] = actions_clipped
        if means_clipped is not None:
            destination.attrs["means_clipped"] = means_clipped
        for episode_id in episodes:
            path = input_dir / f"{task}_episode_{episode_id}.h5"
            with h5py.File(path, "r") as source:
                signals = read_recording(source, selected, len(axes["joint_names"]))
                write_episode(destination, episode_id, signals, {"muscle_controls": "before_step"})
    return output.stat().st_size


def recording_axes(input_dir, task, episode):
    """Use recorded column names, falling back to the current task model."""
    with h5py.File(input_dir / f"{task}_episode_{episode}.h5", "r") as file:
        if "joint_names" in file and "muscle_names" in file:
            return {name: file[name].asstr()[:].tolist()
                    for name in ("joint_names", "muscle_names")}
    from definitions import ENV_CONFIG_PATH
    from envs.environment_factory import EnvironmentFactory

    config = json.loads((Path(ENV_CONFIG_PATH) / f"arnold_{task}_config.json").read_text())
    env = EnvironmentFactory.create(**config)
    try:
        return {"joint_names": list(env.unwrapped.joint_names),
                "muscle_names": list(env.unwrapped.muscle_names)}
    finally:
        env.close()


def main():
    args = parse_args()
    episodes = args.episodes if args.episodes is not None else range(args.num_episodes)
    axes = (json.loads(args.axes.read_text()) if args.axes else
            recording_axes(args.input_dir, args.task, episodes[0]))
    size = export_recordings(args.input_dir, args.output, args.task, args.policy_id,
                             episodes, args.signals, axes, args.horizon, args.dt, args.actions_clipped, args.means_clipped)
    print(f"Saved {len(episodes)} episodes ({size:,} bytes) to {args.output}")


if __name__ == "__main__":
    main()
