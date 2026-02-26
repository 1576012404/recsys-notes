from __future__ import annotations

from typing import List

import torch
from torch import nn

from .common import DenseLinear, FeatureEmbedding, SparseLinear, fm_interaction


class FMModel(nn.Module):
    def __init__(self, field_dims: List[int], dense_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.sparse_linear = SparseLinear(field_dims)
        self.dense_linear = DenseLinear(dense_dim)
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)

    def forward(self, x_dense: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        linear_term = self.sparse_linear(x_sparse) + self.dense_linear(x_dense)
        interaction = fm_interaction(self.embedding(x_sparse))
        return linear_term + interaction

