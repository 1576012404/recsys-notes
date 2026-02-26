from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

DENSE_FIELDS = 13
SPARSE_FIELDS = 26
TOTAL_FIELDS = 1 + DENSE_FIELDS + SPARSE_FIELDS


def _parse_sparse_id(val: str) -> int:
    if not val:
        return 0
    try:
        return int(float(val))
    except ValueError as exc:
        raise ValueError(f"Sparse feature value '{val}' is not numeric. Expected pre-encoded IDs.") from exc


def _parse_parts(parts: List[str]) -> Tuple[int, List[float], List[int]]:
    if len(parts) < TOTAL_FIELDS:
        parts = parts + [""] * (TOTAL_FIELDS - len(parts))
    label = int(parts[0])
    dense_vals = []
    for val in parts[1 : 1 + DENSE_FIELDS]:
        if not val:
            dense_vals.append(0.0)
        else:
            dense_vals.append(float(val))
    sparse_vals = [_parse_sparse_id(x) for x in parts[1 + DENSE_FIELDS : 1 + DENSE_FIELDS + SPARSE_FIELDS]]
    return label, dense_vals, sparse_vals


def _iter_records(
    path: Path,
    delimiter: str = ",",
    has_header: bool = True,
    limit: Optional[int] = None,
) -> Iterable[Tuple[int, List[float], List[int]]]:
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
    max_rows: Optional[int] = None,
    dense_transform: str = "none",
    seed: int = 2026,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(raw_train_path)
    valid_path = Path(raw_valid_path)
    test_path = Path(raw_test_path)

    def count_rows(path: Path, limit: Optional[int] = None) -> int:
        count = 0
        for _ in tqdm(_iter_records(path, delimiter=delimiter, has_header=has_header, limit=limit), desc=f"Count {path.name}"):
            count += 1
        return count

    field_max = np.zeros((SPARSE_FIELDS,), dtype=np.int64)

    def encode_file(path: Path, desc: str, limit: Optional[int] = None):
        rows = count_rows(path, limit=limit)
        y_np = np.zeros((rows,), dtype=np.float32)
        d_np = np.zeros((rows, DENSE_FIELDS), dtype=np.float32)
        s_np = np.zeros((rows, SPARSE_FIELDS), dtype=np.int32)
        idx = 0
        for label, dense_vals, sparse_vals in tqdm(
            _iter_records(path, delimiter=delimiter, has_header=has_header, limit=limit), desc=desc
        ):
            y_np[idx] = float(label)
            d_np[idx] = np.asarray(dense_vals, dtype=np.float32)
            sparse_arr = np.asarray(sparse_vals, dtype=np.int32)
            s_np[idx] = sparse_arr
            np.maximum(field_max, sparse_arr.astype(np.int64), out=field_max)
            idx += 1
        d_np = _transform_dense(d_np, dense_transform).astype(np.float32)
        return y_np, d_np, s_np

    train_y, train_dense, train_sparse = encode_file(train_path, "Encode train", limit=max_rows)
    valid_y, valid_dense, valid_sparse = encode_file(valid_path, "Encode valid")
    test_y, test_dense, test_sparse = encode_file(test_path, "Encode test")
    split_data = {
        "train": (train_y, train_dense, train_sparse),
        "valid": (valid_y, valid_dense, valid_sparse),
        "test": (test_y, test_dense, test_sparse),
    }
    total_rows = int(train_y.shape[0] + valid_y.shape[0] + test_y.shape[0])
    field_dims = [int(field_max[i]) + 1 for i in range(SPARSE_FIELDS)]

    np.savez(
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
        "sparse_id_mode": "pre_encoded",
        "dense_transform": dense_transform,
        "split": {"train": "from_file", "valid": "from_file", "test": "from_file"},
        "delimiter": delimiter,
        "has_header": has_header,
        "seed": seed,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
