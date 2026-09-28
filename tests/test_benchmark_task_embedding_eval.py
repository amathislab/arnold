#!/usr/bin/env python3
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eval_task_utils import (  # noqa: E402
    add_task_embedding_env_id,
    get_task_embedding_index,
    unique_preserve_order,
)


class DummyFeatureExtractor:
    def __init__(self, position_embedding, task_names=None):
        self.position_embedding = position_embedding
        self.task_names = task_names


class DummyPolicy:
    def __init__(self, position_embedding, task_names=None):
        self.features_extractor = DummyFeatureExtractor(position_embedding, task_names)


def test_unique_preserve_order_keeps_first_occurrences():
    tasks = ["kinesis", "relocate", "kinesis", "pen", "relocate"]

    assert unique_preserve_order(tasks) == ["kinesis", "relocate", "pen"]


def test_task_embedding_index_uses_training_config_index():
    policy = DummyPolicy(
        "task_specific",
        ["hand_thumb_reach", "kinesis", "relocate", "kinesis"],
    )

    assert get_task_embedding_index(policy, "kinesis", []) == 1


def test_task_embedding_index_is_none_for_shared_embeddings():
    policy = DummyPolicy("learned", ["hand_thumb_reach"])

    assert get_task_embedding_index(policy, "hand_thumb_reach", []) is None


def test_task_embedding_index_rejects_untrained_task():
    policy = DummyPolicy("task_specific", ["hand_thumb_reach"])

    with pytest.raises(ValueError, match="was not trained with task"):
        get_task_embedding_index(policy, "pen", [])


def test_add_task_embedding_env_id_matches_vectorized_env_shape():
    observation = {"obs": np.zeros((2, 3, 4), dtype=np.float32)}

    updated = add_task_embedding_env_id(observation, 7)

    assert updated["env_id"].shape == (2, 1, 1)
    assert updated["env_id"].dtype == np.int32
    assert (updated["env_id"] == 7).all()


def test_add_task_embedding_env_id_is_noop_without_task_specific_index():
    observation = {"obs": np.zeros((1, 3, 4), dtype=np.float32)}

    updated = add_task_embedding_env_id(observation, None)

    assert "env_id" not in updated
