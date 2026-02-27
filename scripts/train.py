from __future__ import annotations

import argparse
from pathlib import Path

from scripts._common import ROOT, load_train_config, print_json

import sys

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.train.runner import train_once  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model run.")
    parser.add_argument("--model", required=True, choices=["lr", "fm", "fnn", "dnn", "wide_deep", "deepfm"])
    parser.add_argument("--common-config", default="configs/models/common.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrain-checkpoint", default=None)
    parser.add_argument("--pretrain-run-id", default=None)
    parser.add_argument("--debug", action="store_true", help="Quick debug mode: fewer steps and eval batches.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Limit steps per epoch for fast iteration.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Limit eval batches for fast iteration.")
    parser.add_argument(
        "--no-train-metrics",
        action="store_true",
        help="Skip train AUC/LogLoss calculation each epoch (faster).",
    )
    args = parser.parse_args()

    model_cfg = args.model_config or f"configs/models/{args.model}.yaml"
    overrides = {"model": args.model}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.pretrain_checkpoint:
        overrides["pretrain_checkpoint"] = args.pretrain_checkpoint
    if args.pretrain_run_id:
        overrides["pretrain_run_id"] = args.pretrain_run_id
        overrides["init_from"] = "fm_pretrain"
    if args.max_train_steps is not None:
        overrides["max_train_steps_per_epoch"] = args.max_train_steps
    if args.max_eval_batches is not None:
        overrides["max_eval_batches"] = args.max_eval_batches
    if args.no_train_metrics:
        overrides["compute_train_metrics"] = False
    if args.debug:
        overrides["epochs"] = min(int(overrides.get("epochs", 20)), 3)
        overrides["max_train_steps_per_epoch"] = int(overrides.get("max_train_steps_per_epoch", 200))
        overrides["max_eval_batches"] = int(overrides.get("max_eval_batches", 50))
        overrides["compute_train_metrics"] = False

    cfg = load_train_config(args.common_config, model_cfg, overrides)
    summary = train_once(cfg)
    print_json(summary)


if __name__ == "__main__":
    main()
