"""Read task labels and benchmark paths from the release JSON files."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def task_display_names(multiline=False):
    names = json.loads((REPO_ROOT / 'data/reproduction/tasks.json').read_text())
    if multiline:
        return {task: name.replace(' ', '\n', 1) for task, name in names.items()}
    return names


def benchmark_result_paths(method, root=None):
    root = Path(root) if root is not None else REPO_ROOT
    data = json.loads((REPO_ROOT / 'data/reproduction/policies.json').read_text())
    return [str(root / path) for path in data[method.replace('-', '_')]]
