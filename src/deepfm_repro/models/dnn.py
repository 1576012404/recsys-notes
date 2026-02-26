from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

from .common import FeatureEmbedding, MLP


class DNNModel(nn.Module):
    def __init__(
        self,
        field_dims: List[int],
        dense_dim: int,
        embedding_dim: int,
        mlp_dims: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = FeatureEmbedding(field_dims, embedding_dim)
        input_dim = len(field_dims) * embedding_dim + dense_dim
        self.mlp = MLP(input_dim, mlp_dims, dropout=dropout, out_dim=1)

    def forward(self, x_dense: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x_sparse).flatten(start_dim=1)
        x = torch.cat([x_dense, emb], dim=1)
        return self.mlp(x)

