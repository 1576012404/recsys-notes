from __future__ import annotations

import torch

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.models.fm import FMModel  # noqa: E402
from deepfm_repro.models.fnn import FNNModel  # noqa: E402


def test_fnn_init_from_fm_embeddings() -> None:
    field_dims = [4, 5, 6]
    emb_dim = 8
    fm = FMModel(field_dims=field_dims, dense_dim=13, embedding_dim=emb_dim)
    fnn = FNNModel(field_dims=field_dims, dense_dim=13, embedding_dim=emb_dim, mlp_dims=[16, 8], dropout=0.0)

    for i, emb in enumerate(fm.embedding.embeddings):
        emb.weight.data.fill_(i + 1.0)

    fnn.init_from_fm_state(fm.state_dict())
    for i, emb in enumerate(fnn.embedding.embeddings):
        assert torch.allclose(emb.weight, torch.full_like(emb.weight, float(i + 1)))

