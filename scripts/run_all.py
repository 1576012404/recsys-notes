from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full baseline suite.")
    parser.add_argument("--seeds", nargs="+", default=["2026", "2027", "2028"])
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_prepare:
        run([py, "-m", "scripts.prepare_data", "--config", "configs/data.yaml"])

    models = ["lr", "fm", "dnn", "wide_deep", "deepfm"]
    for seed in args.seeds:
        for m in models:
            run([py, "-m", "scripts.train", "--model", m, "--seed", str(seed)])
        run([py, "-m", "scripts.train_fm_for_fnn_init", "--seed", str(seed)])
        run([py, "-m", "scripts.train_fnn", "--seed", str(seed)])

    run([py, "-m", "scripts.compare", "--input", "results/metrics.csv", "--output", "report/comparison.md"])


if __name__ == "__main__":
    main()

