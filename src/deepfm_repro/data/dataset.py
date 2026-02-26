from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class CTRDataset(Dataset):
    def __init__(self, y: np.ndarray, dense: np.ndarray, sparse: np.ndarray) -> None:
        self.y = torch.from_numpy(y.astype(np.float32))
        self.dense = torch.from_numpy(dense.astype(np.float32))
        self.sparse = torch.from_numpy(sparse.astype(np.int64))

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.dense[idx], self.sparse[idx], self.y[idx]


def load_processed_data(npz_path: str | Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    with np.load(npz_path) as data:
        return {
            "train": (data["y_train"], data["dense_train"], data["sparse_train"]),
            "valid": (data["y_valid"], data["dense_valid"], data["sparse_valid"]),
            "test": (data["y_test"], data["dense_test"], data["sparse_test"]),
        }


def load_metadata(metadata_json: str | Path) -> Dict:
    with Path(metadata_json).open("r", encoding="utf-8") as f:
        return json.load(f)

