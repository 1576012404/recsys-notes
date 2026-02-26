from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class SparseLinear(nn.Module):
    def __init__(self, field_dims: List[int]) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, 1) for dim in field_dims])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_sparse: torch.Tensor) -> torch.Tensor:
        out = self.bias.expand(x_sparse.shape[0], 1)
        for i, emb in enumerate(self.embeddings):
            out = out + emb(x_sparse[:, i])
        return out


class DenseLinear(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x_dense: torch.Tensor) -> torch.Tensor:
        return self.linear(x_dense)


class FeatureEmbedding(nn.Module):
    def __init__(self, field_dims: List[int], emb_dim: int) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, emb_dim) for dim in field_dims])

    def forward(self, x_sparse: torch.Tensor) -> torch.Tensor:
        return torch.stack([emb(x_sparse[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)


def fm_interaction(x_embed: torch.Tensor) -> torch.Tensor:
    summed = torch.sum(x_embed, dim=1)
    squared_sum = summed * summed
    sum_squared = torch.sum(x_embed * x_embed, dim=1)
    return 0.5 * torch.sum(squared_sum - sum_squared, dim=1, keepdim=True)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Iterable[int], dropout: float = 0.0, out_dim: int = 1) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        current = in_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(current, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current = hidden
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

