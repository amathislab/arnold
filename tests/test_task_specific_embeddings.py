import os
import sys

import pytest
import torch
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.feature_extractors import TransformerFeaturesExtractor  # noqa: E402
from models.task_specific_embeddings import TaskSpecificPositionalEncoding  # noqa: E402


def test_mixed_task_batch_uses_each_tasks_embedding_for_senses_and_actions():
    observation_space = spaces.Dict({
        "obs": spaces.Box(-1, 1, shape=(2, 5)),
        "obs_ids": spaces.Box(0, 3, shape=(2, 2), dtype=int),
        "action_ids": spaces.Box(0, 3, shape=(1, 2), dtype=int),
    })
    extractor = TransformerFeaturesExtractor(
        observation_space,
        embedding_size=4,
        num_layers=0,
        num_heads=1,
        dropout=0,
        position_embedding="task_specific",
        task_names=["reach", "pen"],
        vocabulary={"value": 0, "sense": 1, "motor": 2, "padding": 3},
        device="cpu",
    )
    with torch.no_grad():
        extractor.positional_encoder.task_embeddings["reach"].weight[:3].fill_(1)
        extractor.positional_encoder.task_embeddings["pen"].weight[:3].fill_(3)

    observation = {
        "obs": torch.zeros(2, 2, 5),
        "obs_ids": torch.tensor([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]),
        "action_ids": torch.tensor([[[1, 2]], [[1, 2]]]),
        "env_id": torch.tensor([[[0]], [[1]]]),
    }
    sensory, _, motor, _, value = extractor(observation)

    torch.testing.assert_close(sensory[1] - sensory[0], torch.full((2, 4), 4.0))
    torch.testing.assert_close(motor[1] - motor[0], torch.full((1, 4), 4.0))
    torch.testing.assert_close(value[1] - value[0], torch.full((1, 4), 2.0))


def test_duplicate_training_entries_share_a_table_and_frozen_tables_stay_frozen():
    encoder = TaskSpecificPositionalEncoding(
        4, 2, ["reach", "pen", "reach"], padding_idx=3,
        learnable=False, dropout=0,
    )
    assert set(encoder.task_embeddings) == {"reach", "pen"}
    result = encoder(torch.zeros(3, 1, 2), torch.tensor([0]), torch.tensor([0, 1, 2]))
    torch.testing.assert_close(result[0], result[2])
    assert all(not table.weight.requires_grad for table in encoder.task_embeddings.values())
    assert all((table.weight[3] == 0).all() for table in encoder.task_embeddings.values())


def test_task_specific_embedding_rejects_invalid_task_index():
    encoder = TaskSpecificPositionalEncoding(4, 2, ["reach"], dropout=0)
    with pytest.raises(ValueError, match="outside the training task list"):
        encoder(torch.zeros(1, 1, 2), torch.tensor([0]), torch.tensor([1]))


def test_mixed_task_gradients_update_only_the_selected_tokens():
    encoder = TaskSpecificPositionalEncoding(4, 2, ["reach", "pen"], dropout=0)
    values = encoder(
        torch.zeros(2, 1, 2),
        torch.tensor([[[1]], [[2]]]),
        torch.tensor([0, 1]),
    )
    values.sum().backward()

    reach_grad = encoder.task_embeddings["reach"].weight.grad
    pen_grad = encoder.task_embeddings["pen"].weight.grad
    assert torch.count_nonzero(reach_grad[1]) == 2
    assert torch.count_nonzero(pen_grad[2]) == 2
    assert torch.count_nonzero(reach_grad[2]) == 0
    assert torch.count_nonzero(pen_grad[1]) == 0


def test_frozen_tables_follow_experiment_seed():
    torch.manual_seed(1)
    first = TaskSpecificPositionalEncoding(4, 2, ["reach"], learnable=False)
    torch.manual_seed(1)
    repeat = TaskSpecificPositionalEncoding(4, 2, ["reach"], learnable=False)
    torch.manual_seed(2)
    different = TaskSpecificPositionalEncoding(4, 2, ["reach"], learnable=False)

    first_weights = first.task_embeddings["reach"].weight
    torch.testing.assert_close(first_weights, repeat.task_embeddings["reach"].weight)
    assert not torch.equal(first_weights, different.task_embeddings["reach"].weight)
