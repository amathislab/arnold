"""Independent positional embeddings for each task."""
import torch
from torch import nn


class TaskSpecificPositionalEncoding(nn.Module):
    def __init__(
        self, num_tokens, d_model, task_names, dropout=0.1,
        padding_idx=None, learnable=True,
    ):
        super().__init__()
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
        for index, embedding in enumerate(self.task_embeddings.values()):
            generator = torch.Generator().manual_seed(42 + index)
            nn.init.normal_(embedding.weight, mean=0, std=0.1, generator=generator)
            if padding_idx is not None:
                embedding.weight.data[padding_idx].zero_()
            embedding.weight.requires_grad_(learnable)

    def forward(self, x, positions=None, task_name=None, task_ids=None):
        if positions is None:
            positions = torch.arange(self.num_tokens, device=x.device)
        positions = positions.long()
        if task_ids is not None:
            result = torch.empty_like(x)
            for index in torch.unique(task_ids):
                mask = task_ids == index
                selected = positions
                if positions.ndim == x.ndim:
                    selected = positions[mask]
                result[mask] = self.forward(
                    x[mask], selected, task_name=self.task_names[int(index)]
                )
            return result
        embedding = self.task_embeddings[task_name](positions)
        if positions.ndim == x.ndim:
            embedding = embedding.sum(dim=-2)
        return self.dropout(x + embedding)
