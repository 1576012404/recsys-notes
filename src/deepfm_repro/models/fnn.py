from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn

from .common import FeatureEmbedding, MLP


class FNNModel(nn.Module):
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

    def init_from_fm_state(self, fm_state_dict: dict) -> None:
        # Copy sparse embedding weights from FM pretraining to FNN embedding layers.
        for i, emb in enumerate(self.embedding.embeddings):
            key = f"embedding.embeddings.{i}.weight"
            if key in fm_state_dict:
                if emb.weight.shape != fm_state_dict[key].shape:
                    raise ValueError(
                        f"Shape mismatch for field {i}: {emb.weight.shape} vs {fm_state_dict[key].shape}"
                    )
                emb.weight.data.copy_(fm_state_dict[key].data)

    def forward(self, x_dense: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x_sparse).flatten(start_dim=1)
        x = torch.cat([x_dense, emb], dim=1)
        return self.mlp(x)

