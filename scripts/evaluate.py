from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import ROOT, print_json

import sys

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.train.runner import evaluate_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-npz", default="data/processed/criteo.npz")
    parser.add_argument("--metadata-json", default="data/processed/metadata.json")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    metrics = evaluate_run(
        run_dir=Path(args.run_dir),
        data_npz=args.data_npz,
        metadata_json=args.metadata_json,
        batch_size=args.batch_size,
    )
    print_json(metrics)


if __name__ == "__main__":
    main()

