from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.utils import merge_dict, read_yaml  # noqa: E402


def load_train_config(common_path: str, model_path: str, overrides: Optional[Dict] = None) -> Dict:
    common = read_yaml(common_path)
    model = read_yaml(model_path)
    return merge_dict(common, model, overrides or {})


def print_json(payload: Dict) -> None:
    print(json.dumps(payload, indent=2))

