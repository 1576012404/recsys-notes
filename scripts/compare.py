from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def _load_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _group(rows, split: str):
    bucket = defaultdict(list)
    for r in rows:
        if r["split"] == split:
            bucket[r["model"]].append({"auc": float(r["auc"]), "logloss": float(r["logloss"])})
    return bucket


def _stat(values):
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model comparison.")
    parser.add_argument("--input", default="results/metrics.csv")
    parser.add_argument("--output", default="report/comparison.md")
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    args = parser.parse_args()

    rows = list(_load_rows(Path(args.input)))
    grouped = _group(rows, args.split)
    ordered = ["lr", "fm", "fnn", "dnn", "wide_deep", "deepfm"]

    lines = [f"# Model Comparison ({args.split})", "", "| Model | AUC (mean+-std) | LogLoss (mean+-std) |", "|---|---:|---:|"]
    stats = {}
    for model in ordered:
        vals = grouped.get(model, [])
        auc_m, auc_s = _stat([v["auc"] for v in vals])
        ll_m, ll_s = _stat([v["logloss"] for v in vals])
        stats[model] = (auc_m, ll_m)
        lines.append(f"| {model} | {auc_m:.6f} +- {auc_s:.6f} | {ll_m:.6f} +- {ll_s:.6f} |")

    deep_auc, deep_ll = stats.get("deepfm", (0.0, 0.0))
    lines.extend(["", "## DeepFM Delta"])
    for model in ["lr", "fm", "fnn", "dnn", "wide_deep"]:
        auc, ll = stats.get(model, (0.0, 0.0))
        lines.append(f"- vs {model}: delta_auc={deep_auc - auc:+.6f}, delta_logloss={deep_ll - ll:+.6f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

