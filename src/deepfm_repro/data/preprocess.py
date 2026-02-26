from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

DENSE_FIELDS = 13
SPARSE_FIELDS = 26
TOTAL_FIELDS = 1 + DENSE_FIELDS + SPARSE_FIELDS


def _parse_parts(parts: List[str]) -> Tuple[int, List[float], List[str]]:
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


def _iter_records(
    path: Path,
    delimiter: str = ",",
    has_header: bool = True,
    limit: Optional[int] = None,
) -> Iterable[Tuple[int, List[float], List[str]]]:
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
        if not first:
            return
        first_parts = first.rstrip("\n").split(delimiter)
        count = 0
        if not has_header:
            yield _parse_parts(first_parts)
            count += 1
        for line in f:
            if limit is not None and count >= limit:
                break
            parts = line.rstrip("\n").split(delimiter)
            yield _parse_parts(parts)
            count += 1


def _transform_dense(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    if mode == "none":
        return values
    raise ValueError(f"Unsupported dense_transform: {mode}")


def preprocess_criteo(
    raw_train_path: str | Path,
    raw_valid_path: str | Path,
    raw_test_path: str | Path,
    output_dir: str | Path,
    delimiter: str = ",",
    has_header: bool = True,
    min_freq: int = 10,
    max_rows: Optional[int] = None,
    dense_transform: str = "log1p",
    seed: int = 2026,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(raw_train_path)
    valid_path = Path(raw_valid_path)
    test_path = Path(raw_test_path)

    sparse_counters = [Counter() for _ in range(SPARSE_FIELDS)]
    total_rows = 0
    for _, _, sparse_vals in tqdm(
        _iter_records(train_path, delimiter=delimiter, has_header=has_header, limit=max_rows),
        desc="Pass1 counting vocab(train)",
    ):
        for i, token in enumerate(sparse_vals):
            sparse_counters[i][token] += 1
        total_rows += 1

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

    def encode_file(path: Path, desc: str, limit: Optional[int] = None):
        ys: List[float] = []
        ds: List[List[float]] = []
        ss: List[List[int]] = []
        for label, dense_vals, sparse_vals in tqdm(
            _iter_records(path, delimiter=delimiter, has_header=has_header, limit=limit), desc=desc
        ):
            ys.append(float(label))
            ds.append(dense_vals)
            ss.append([vocab_maps[i].get(token, 0) for i, token in enumerate(sparse_vals)])
        y_np = np.array(ys, dtype=np.float32)
        d_np = _transform_dense(np.array(ds, dtype=np.float32), dense_transform).astype(np.float32)
        s_np = np.array(ss, dtype=np.int64)
        return y_np, d_np, s_np

    train_y, train_dense, train_sparse = encode_file(train_path, "Pass2 encoding train", limit=max_rows)
    valid_y, valid_dense, valid_sparse = encode_file(valid_path, "Pass2 encoding valid")
    test_y, test_dense, test_sparse = encode_file(test_path, "Pass2 encoding test")
    split_data = {
        "train": (train_y, train_dense, train_sparse),
        "valid": (valid_y, valid_dense, valid_sparse),
        "test": (test_y, test_dense, test_sparse),
    }
    total_rows = int(train_y.shape[0] + valid_y.shape[0] + test_y.shape[0])

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
        "raw_train_path": str(train_path),
        "raw_valid_path": str(valid_path),
        "raw_test_path": str(test_path),
        "rows": total_rows,
        "dense_fields": DENSE_FIELDS,
        "sparse_fields": SPARSE_FIELDS,
        "field_dims": field_dims,
        "min_freq": min_freq,
        "dense_transform": dense_transform,
        "split": {"train": "from_file", "valid": "from_file", "test": "from_file"},
        "delimiter": delimiter,
        "has_header": has_header,
        "seed": seed,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
