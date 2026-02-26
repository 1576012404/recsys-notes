from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

DENSE_FIELDS = 13
SPARSE_FIELDS = 26
TOTAL_FIELDS = 1 + DENSE_FIELDS + SPARSE_FIELDS


def _parse_line(line: str) -> Tuple[int, List[float], List[str]]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < TOTAL_FIELDS:
        parts = parts + [""] * (TOTAL_FIELDS - len(parts))
    label = int(parts[0])
    dense_vals = []
    for val in parts[1 : 1 + DENSE_FIELDS]:
        if not val:
            dense_vals.append(0.0)
        else:
            dense_vals.append(float(val))
    sparse_vals = [x if x else "UNK" for x in parts[1 + DENSE_FIELDS : 1 + DENSE_FIELDS + SPARSE_FIELDS]]
    return label, dense_vals, sparse_vals


def _transform_dense(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    if mode == "none":
        return values
    raise ValueError(f"Unsupported dense_transform: {mode}")


def preprocess_criteo(
    raw_path: str | Path,
    output_dir: str | Path,
    min_freq: int = 10,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_rows: Optional[int] = None,
    dense_transform: str = "log1p",
    seed: int = 2026,
) -> None:
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    raw_path = Path(raw_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse_counters = [Counter() for _ in range(SPARSE_FIELDS)]
    total_rows = 0
    with raw_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Pass1 counting vocab"):
            _, _, sparse = _parse_line(line)
            for i, token in enumerate(sparse):
                sparse_counters[i][token] += 1
            total_rows += 1
            if max_rows and total_rows >= max_rows:
                break

    vocab_maps: List[Dict[str, int]] = []
    field_dims: List[int] = []
    for counter in sparse_counters:
        mapping = {"UNK": 0}
        for token, count in counter.items():
            if token == "UNK":
                continue
            if count >= min_freq:
                mapping[token] = len(mapping)
        vocab_maps.append(mapping)
        field_dims.append(len(mapping))

    y = np.zeros((total_rows,), dtype=np.float32)
    dense = np.zeros((total_rows, DENSE_FIELDS), dtype=np.float32)
    sparse = np.zeros((total_rows, SPARSE_FIELDS), dtype=np.int64)

    with raw_path.open("r", encoding="utf-8") as f:
        idx = 0
        for line in tqdm(f, desc="Pass2 encoding"):
            label, dense_vals, sparse_vals = _parse_line(line)
            y[idx] = label
            dense[idx] = dense_vals
            for i, token in enumerate(sparse_vals):
                sparse[idx, i] = vocab_maps[i].get(token, 0)
            idx += 1
            if idx >= total_rows:
                break

    dense = _transform_dense(dense, dense_transform).astype(np.float32)

    rng = np.random.default_rng(seed)
    indices = np.arange(total_rows)
    rng.shuffle(indices)
    y = y[indices]
    dense = dense[indices]
    sparse = sparse[indices]

    train_end = int(total_rows * train_ratio)
    valid_end = train_end + int(total_rows * valid_ratio)

    split_data = {
        "train": (y[:train_end], dense[:train_end], sparse[:train_end]),
        "valid": (y[train_end:valid_end], dense[train_end:valid_end], sparse[train_end:valid_end]),
        "test": (y[valid_end:], dense[valid_end:], sparse[valid_end:]),
    }

    np.savez_compressed(
        output_dir / "criteo.npz",
        y_train=split_data["train"][0],
        dense_train=split_data["train"][1],
        sparse_train=split_data["train"][2],
        y_valid=split_data["valid"][0],
        dense_valid=split_data["valid"][1],
        sparse_valid=split_data["valid"][2],
        y_test=split_data["test"][0],
        dense_test=split_data["test"][1],
        sparse_test=split_data["test"][2],
    )

    metadata = {
        "raw_path": str(raw_path),
        "rows": total_rows,
        "dense_fields": DENSE_FIELDS,
        "sparse_fields": SPARSE_FIELDS,
        "field_dims": field_dims,
        "min_freq": min_freq,
        "dense_transform": dense_transform,
        "split": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "seed": seed,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

