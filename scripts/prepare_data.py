from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.data import preprocess_criteo  # noqa: E402
from deepfm_repro.utils import read_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Criteo dataset.")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    preprocess_criteo(
        raw_train_path=cfg["raw_train_path"],
        raw_valid_path=cfg["raw_valid_path"],
        raw_test_path=cfg["raw_test_path"],
        output_dir=cfg["output_dir"],
        delimiter=cfg.get("delimiter", ","),
        has_header=bool(cfg.get("has_header", True)),
        min_freq=int(cfg.get("min_freq", 10)),
        max_rows=cfg.get("max_rows"),
        dense_transform=cfg.get("dense_transform", "log1p"),
        seed=int(cfg.get("seed", 2026)),
    )


if __name__ == "__main__":
    main()
