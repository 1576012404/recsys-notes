from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

from .common import DenseLinear, FeatureEmbedding, MLP, SparseLinear


class WideDeepModel(nn.Module):
    def __init__(
        self,
        field_dims: List[int],
        dense_dim: int,
        embedding_dim: int,
        mlp_dims: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.sparse_linear = SparseLinear(field_dims)
        self.dense_linear = DenseLinear(dense_dim)
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        input_dim = len(field_dims) * embedding_dim + dense_dim
        self.deep = MLP(input_dim, mlp_dims, dropout=dropout, out_dim=1)

    def forward(self, x_dense: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        wide = self.sparse_linear(x_sparse) + self.dense_linear(x_dense)
        deep_in = torch.cat([x_dense, self.embedding(x_sparse).flatten(start_dim=1)], dim=1)
        deep = self.deep(deep_in)
        return wide + deep

