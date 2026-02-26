from __future__ import annotations

from typing import List

import torch
from torch import nn

from .common import DenseLinear, SparseLinear


class LRModel(nn.Module):
    def __init__(self, field_dims: List[int], dense_dim: int) -> None:
        super().__init__()
        self.sparse_linear = SparseLinear(field_dims)
        self.dense_linear = DenseLinear(dense_dim)

    def forward(self, x_dense: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        return self.sparse_linear(x_sparse) + self.dense_linear(x_dense)

