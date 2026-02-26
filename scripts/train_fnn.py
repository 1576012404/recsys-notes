from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts._common import load_train_config, print_json

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.train.runner import train_once  # noqa: E402


def _resolve_fm_run(results_dir: Path, run_id: str | None) -> tuple[str, Path]:
    runs_dir = results_dir / "runs"
    if run_id:
        ckpt = runs_dir / run_id / "best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"FM checkpoint not found: {ckpt}")
        return run_id, ckpt

    metrics_csv = results_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError("metrics.csv not found; run `train_fm_for_fnn_init` first.")

    latest_run = None
    with metrics_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model") == "fm":
                latest_run = row["run_id"]
    if not latest_run:
        raise RuntimeError("No FM runs found in metrics.csv")
    ckpt = runs_dir / latest_run / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"FM checkpoint missing for run_id={latest_run}")
    return latest_run, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FNN initialized by FM embeddings.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fm-run-id", default=None)
    parser.add_argument("--common-config", default="configs/models/common.yaml")
    parser.add_argument("--model-config", default="configs/models/fnn.yaml")
    args = parser.parse_args()

    common = load_train_config(args.common_config, args.model_config, {"model": "fnn"})
    results_dir = Path(common.get("results_dir", "results"))
    pretrain_run_id, pretrain_ckpt = _resolve_fm_run(results_dir, args.fm_run_id)

    overrides = {
        "model": "fnn",
        "init_from": "fm_pretrain",
        "pretrain_run_id": pretrain_run_id,
        "pretrain_checkpoint": str(pretrain_ckpt),
    }
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = load_train_config(args.common_config, args.model_config, overrides)
    summary = train_once(cfg)
    print_json(summary)


if __name__ == "__main__":
    main()

