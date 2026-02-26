from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def read_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dict(*items: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for item in items:
        merged.update(item)
    return merged


def config_hash(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

