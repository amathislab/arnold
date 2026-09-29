import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


def unique_preserve_order(values: List[str]) -> List[str]:
    """Return unique values without changing their first-seen order."""
    seen = set()
    unique_values = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def load_training_args(policy_path: Optional[str]) -> Dict[str, Any]:
    """Load args.json from the policy directory if it is available."""
    if policy_path is None:
        return {}

    policy_dir = os.path.dirname(policy_path)
    args_path = os.path.join(policy_dir, "args.json")

    if not os.path.exists(args_path):
        return {}

    with open(args_path, "r") as f:
        return json.load(f)


def get_tasks_from_args_json(policy_path):
    """Get the unique task list and memory length from args.json."""
    training_args = load_training_args(policy_path)
    tasks = training_args.get("tasks")
    if tasks is None:
        return None, None
    return unique_preserve_order(tasks), training_args.get("num_memory_steps")


def get_task_embedding_index(
    policy, task_name: str, training_tasks: List[str]
) -> Optional[int]:
    """Return the training config index used by task-specific embeddings.

    FlexibleMultiVecNormalize reindexes the single eval env to row 0, but the
    task-specific positional encoder expects the original training config index
    in observation["env_id"]. Keep those two ids separate.
    """
    feature_extractor = getattr(policy, "features_extractor", None)
    if getattr(feature_extractor, "position_embedding", None) != "task_specific":
        return None

    task_names = getattr(feature_extractor, "task_names", None) or training_tasks
    if not task_names:
        raise ValueError(
            "Loaded task-specific policy does not expose task_names and no "
            "training args.json task list was found."
        )

    try:
        return task_names.index(task_name)
    except ValueError as exc:
        raise ValueError(
            f"Task-specific policy was not trained with task '{task_name}'. "
            f"Available tasks: {unique_preserve_order(task_names)}"
        ) from exc


def add_task_embedding_env_id(
    observation: Dict[str, np.ndarray], task_embedding_index: Optional[int]
) -> Dict[str, np.ndarray]:
    """Add the training task index expected by task-specific embeddings."""
    if task_embedding_index is None:
        return observation

    batch_size = next(iter(observation.values())).shape[0]
    observation["env_id"] = np.full(
        (batch_size, 1, 1), task_embedding_index, dtype=np.int32
    )
    return observation
