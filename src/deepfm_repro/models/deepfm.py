from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

from .common import DenseLinear, FeatureEmbedding, MLP, SparseLinear, fm_interaction


class DeepFMModel(nn.Module):
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
        deep_in_dim = len(field_dims) * embedding_dim + dense_dim
        self.deep = MLP(deep_in_dim, mlp_dims, dropout=dropout, out_dim=1)

    def forward(self, x_dense: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x_sparse)
        linear = self.sparse_linear(x_sparse) + self.dense_linear(x_dense)
        fm = fm_interaction(emb)
        deep = self.deep(torch.cat([x_dense, emb.flatten(start_dim=1)], dim=1))
        return linear + fm + deep

