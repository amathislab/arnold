"""Read task labels and benchmark paths from the release JSON files."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def task_display_names(multiline=False):
    data = json.loads((REPO_ROOT / 'data/reproduction/tasks.json').read_text())
    names = {task: spec['display_name'] for task, spec in data['tasks'].items()}
    if multiline:
        return {task: name.replace(' ', '\n', 1) for task, name in names.items()}
    return names


def benchmark_result_paths(method, root=None):
    root = Path(root) if root is not None else REPO_ROOT
    data = json.loads((REPO_ROOT / 'data/reproduction/policies.json').read_text())
    aliases = {alias: name for name, spec in data['methods'].items() for alias in spec['aliases']}
    method = aliases.get(method, method)
    return [str(root / policy['benchmark']) for policy in data['policies'] if policy['method'] == method]
