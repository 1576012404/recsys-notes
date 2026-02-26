from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.train.runner import _append_metrics  # noqa: E402


def test_metrics_schema_contains_pretrain_columns(tmp_path) -> None:
    csv_path = tmp_path / "metrics.csv"
    _append_metrics(
        csv_path,
        {
            "run_id": "x",
            "model": "fnn",
            "seed": 1,
            "split": "test",
            "auc": 0.5,
            "logloss": 0.6,
            "best_epoch": 1,
            "config_hash": "abc",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "init_from": "fm_pretrain",
            "pretrain_run_id": "fm_1",
        },
    )
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "init_from" in header
    assert "pretrain_run_id" in header

