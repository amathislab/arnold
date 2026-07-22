import torch
import math
from torch import nn
from typing import Optional, Any


class SinCosPositionalEncoding(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(num_tokens).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(num_tokens, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if padding_idx is not None:
            pe[padding_idx, :] = (
                0  # So that the embedding corresponding to the padding idx is 0
            )
        self.register_buffer("embedding", pe)

    def forward(self, x, positions=None):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        if positions is None:
            x = x + self.embedding
        else:
            positions = positions.int()
            if len(positions.shape) == len(x.shape):
                x = x + self.embedding[positions, :].sum(dim=-2)
            else:
                x = x + self.embedding[positions, :]
        return self.dropout(x)

    def set_pretained(self, embeddings):
        raise NotImplementedError()


class LearnedPositionalEncoding(nn.Module):
    def __init__(
        self,
        num_tokens,
        d_model,
        dropout: float = 0.1,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.num_tokens = num_tokens
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=d_model, padding_idx=padding_idx
        )

    def forward(self, x, positions=None):
        if positions is None:
            positions = torch.arange(self.num_tokens, device=x.device)
        else:
            positions = positions.int()
        if len(positions.shape) == len(x.shape):
            # We consider that the last dimension of positions indicates
            # multiple positions for the same input
            x = x + self.embedding(positions).sum(dim=-2)
        else:
            # if len(positions.shape) == len(x.shape) - 1:# Only differs on the batch dimension
            #     x = x + self.embedding(positions).sum(dim=-2).unsqueeze(0)
            if len(positions.shape) == 0 :            # All elements use the same position
                x = x + self.embedding(positions).unsqueeze(0)
            else :
                raise NotImplementedError(f"The positional index and the original vector should have the same size. Get {positions.shape} and {x.shape}")
        return self.dropout(x)

    def set_pretrained(self, embeddings, padding_idx):
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings, freeze=False, padding_idx=padding_idx
        )


class Mean(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)


class ReplicateInputAttentionWrapper(nn.Module):
    def __init__(self, wrapped, n=1):
        super().__init__()
        self.n = n
        self.wrapped = wrapped

    def forward(self, x):
        replicate_x = [x] * self.n
        return self.wrapped(*replicate_x)[0]


class NoisyAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x, value_noise=None):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if value_noise is not None:
            v += value_noise
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out, attn


class Identity(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        return input
