"""Shared display labels and benchmark selections for analysis and plotting."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

TASK_NAMES = {
    "hand_little_reach": "Little reach",
    "hand_index_reach": "Index reach",
    "hand_middle_reach": "Middle reach",
    "hand_ring_reach": "Ring reach",
    "hand_thumb_reach": "Thumb reach",
    "reorient": "Die reorient",
    "pen": "Pen reorient",
    "baoding_p1_cw": "Baoding CW",
    "baoding_p1_ccw": "Baoding CCW",
    "baoding_p2_overlap": "Baoding hard",
    "baoding_p2": "Baoding harder",
    "elbow_pose": "Elbow pose",
    "relocate": "Object relocation",
    "kinesis": "Walk to point"
}
EXTRA_TASK_NAMES = {"hand_pose": "Hand pose", "elbow_joint_pose": "Elbow joint", "finger_pose": "Finger pose"}


def task_display_names(multiline=False, include_extra=False):
    names = dict(TASK_NAMES)
    if include_extra:
        names.update(EXTRA_TASK_NAMES)
    if multiline:
        names = {task: name.rsplit(" ", 1)[0] + "\n" + name.rsplit(" ", 1)[1]
                 for task, name in names.items()}
    return names


def benchmark_result_paths(method, root=None):
    root = Path(root) if root is not None else REPO_ROOT
    selections = json.loads((REPO_ROOT / "data/analysis/reproduction/policies.json").read_text())
    return [str(root / path) for path in selections[method.replace("-", "_")]]
