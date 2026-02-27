from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import List

from scripts._common import ROOT, load_train_config

import sys

SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.train.runner import train_once  # noqa: E402


@dataclass
class TrialResult:
    model: str
    seed: int
    lr: float
    batch_size: int
    eval_batch_size: int
    valid_auc: float
    valid_logloss: float
    test_auc: float
    test_logloss: float
    run_id: str


def _write_csv(path: Path, rows: List[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "seed",
                "learning_rate",
                "batch_size",
                "eval_batch_size",
                "valid_auc",
                "valid_logloss",
                "test_auc",
                "test_logloss",
                "run_id",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.model,
                    r.seed,
                    r.lr,
                    r.batch_size,
                    r.eval_batch_size,
                    f"{r.valid_auc:.6f}",
                    f"{r.valid_logloss:.6f}",
                    f"{r.test_auc:.6f}",
                    f"{r.test_logloss:.6f}",
                    r.run_id,
                ]
            )


def _write_markdown(path: Path, rows: List[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda x: (-x.valid_auc, x.valid_logloss))
    lines = [
        "# Quick Tune Results",
        "",
        "| Rank | Model | LR | Batch | Eval Batch | Seed | Valid AUC | Valid LogLoss | Test AUC | Test LogLoss | Run ID |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for i, r in enumerate(sorted_rows, start=1):
        lines.append(
            f"| {i} | {r.model} | {r.lr:.6g} | {r.batch_size} | {r.eval_batch_size} | {r.seed} | "
            f"{r.valid_auc:.6f} | {r.valid_logloss:.6f} | {r.test_auc:.6f} | {r.test_logloss:.6f} | {r.run_id} |"
        )

    # Mean by hyperparameter tuple across seeds.
    grouped = {}
    for r in rows:
        key = (r.model, r.lr, r.batch_size, r.eval_batch_size)
        grouped.setdefault(key, []).append(r)

    lines.extend(
        [
            "",
            "## Mean Across Seeds",
            "",
            "| Model | LR | Batch | Eval Batch | n | Mean Valid AUC | Mean Valid LogLoss | Mean Test AUC | Mean Test LogLoss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, vals in sorted(grouped.items(), key=lambda kv: -mean([x.valid_auc for x in kv[1]])):
        model, lr, bs, ebs = key
        lines.append(
            f"| {model} | {lr:.6g} | {bs} | {ebs} | {len(vals)} | "
            f"{mean([x.valid_auc for x in vals]):.6f} | {mean([x.valid_logloss for x in vals]):.6f} | "
            f"{mean([x.test_auc for x in vals]):.6f} | {mean([x.test_logloss for x in vals]):.6f} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick hyperparameter tuning with short-budget training.")
    parser.add_argument("--model", default="deepfm", choices=["lr", "fm", "fnn", "dnn", "wide_deep", "deepfm"])
    parser.add_argument("--common-config", default="configs/models/common.yaml")
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.001, 0.0015])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8192])
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--output-csv", default="results/quick_tune.csv")
    parser.add_argument("--output-md", default="report/quick_tune.md")
    args = parser.parse_args()

    model_cfg = args.model_config or f"configs/models/{args.model}.yaml"
    rows: List[TrialResult] = []

    for seed in args.seeds:
        for lr in args.lrs:
            for batch_size in args.batch_sizes:
                overrides = {
                    "model": args.model,
                    "seed": seed,
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "epochs": args.epochs,
                    "max_train_steps_per_epoch": args.max_train_steps,
                    "max_eval_batches": args.max_eval_batches,
                }
                cfg = load_train_config(args.common_config, model_cfg, overrides)
                summary = train_once(cfg)
                rows.append(
                    TrialResult(
                        model=args.model,
                        seed=seed,
                        lr=lr,
                        batch_size=batch_size,
                        eval_batch_size=args.eval_batch_size,
                        valid_auc=float(summary["valid"]["auc"]),
                        valid_logloss=float(summary["valid"]["logloss"]),
                        test_auc=float(summary["test"]["auc"]),
                        test_logloss=float(summary["test"]["logloss"]),
                        run_id=str(summary["run_id"]),
                    )
                )

    _write_csv(Path(args.output_csv), rows)
    _write_markdown(Path(args.output_md), rows)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()

