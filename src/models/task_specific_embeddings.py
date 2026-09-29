"""Independent token embeddings for the task-specific vocabulary ablation."""

import torch
from torch import nn


class TaskSpecificPositionalEncoding(nn.Module):
    """Use the same token IDs with a separate embedding table for each task."""

    def __init__(
        self,
        num_tokens,
        d_model,
        task_names,
        dropout=0.1,
        padding_idx=None,
        learnable=True,
    ):
        super().__init__()
        if not task_names:
            raise ValueError("task_names must contain at least one task")
        self.dropout = nn.Dropout(dropout)
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.task_names = task_names
        self.learnable = learnable
        self.padding_idx = padding_idx
        self.task_embeddings = nn.ModuleDict({
            name: nn.Embedding(num_tokens, d_model, padding_idx=padding_idx)
            for name in dict.fromkeys(task_names)
        })
        for embedding in self.task_embeddings.values():
            nn.init.normal_(embedding.weight, mean=0, std=0.1)
            if padding_idx is not None:
                embedding.weight.data[padding_idx].zero_()
            embedding.weight.requires_grad_(learnable)

    def forward(self, x, positions, task_ids):
        """Select each row's task table, including mixed-task training batches."""
        task_ids = task_ids.reshape(-1).long()
        if task_ids.numel() != x.shape[0]:
            raise ValueError("Expected one task ID per batch row")
        if torch.any(task_ids < 0) or torch.any(task_ids >= len(self.task_names)):
            raise ValueError("Task ID is outside the training task list")

        positions = positions.long()
        result = torch.empty_like(x)
        for index in torch.unique(task_ids).tolist():
            mask = task_ids == index
            selected_positions = positions[mask] if positions.ndim == x.ndim else positions
            embedding = self.task_embeddings[self.task_names[index]](selected_positions)
            if selected_positions.ndim == x.ndim:
                embedding = embedding.sum(dim=-2)
            result[mask] = self.dropout(x[mask] + embedding)
        return result
