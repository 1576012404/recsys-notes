from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.models import DeepFMModel, DNNModel, FMModel, FNNModel, LRModel, WideDeepModel  # noqa: E402


def test_all_models_forward_shape() -> None:
    batch = 4
    dense_dim = 13
    field_dims = [10] * 26
    dense = torch.randn(batch, dense_dim)
    sparse = torch.randint(0, 10, (batch, 26))

    models = [
        LRModel(field_dims, dense_dim),
        FMModel(field_dims, dense_dim, embedding_dim=8),
        FNNModel(field_dims, dense_dim, embedding_dim=8, mlp_dims=[16], dropout=0.1),
        DNNModel(field_dims, dense_dim, embedding_dim=8, mlp_dims=[16], dropout=0.1),
        WideDeepModel(field_dims, dense_dim, embedding_dim=8, mlp_dims=[16], dropout=0.1),
        DeepFMModel(field_dims, dense_dim, embedding_dim=8, mlp_dims=[16], dropout=0.1),
    ]
    for model in models:
        out = model(dense, sparse)
        assert out.shape == (batch, 1)

