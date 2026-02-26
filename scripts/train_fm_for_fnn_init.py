from __future__ import annotations

import argparse

from scripts._common import load_train_config, print_json

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.train.runner import train_once  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FM for FNN initialization.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--common-config", default="configs/models/common.yaml")
    parser.add_argument("--model-config", default="configs/models/fm.yaml")
    args = parser.parse_args()

    overrides = {"model": "fm", "init_from": "none", "pretrain_run_id": ""}
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_train_config(args.common_config, args.model_config, overrides)
    summary = train_once(cfg)
    print_json(summary)


if __name__ == "__main__":
    main()

