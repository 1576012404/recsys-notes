from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from deepfm_repro.data.dataset import load_metadata  # noqa: E402
from deepfm_repro.train.runner import _build_model  # noqa: E402
from deepfm_repro.utils import merge_dict, read_yaml  # noqa: E402


def _count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _group_params(model):
    groups = defaultdict(int)
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        groups[key] += p.numel()
    return dict(sorted(groups.items(), key=lambda x: x[0]))


def _format_num(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Print model structure and parameter counts.")
    parser.add_argument("--models", nargs="+", default=["lr", "fm", "fnn", "dnn", "wide_deep", "deepfm"])
    parser.add_argument("--metadata-json", default="data/processed/metadata.json")
    parser.add_argument("--common-config", default="configs/models/common.yaml")
    parser.add_argument("--dense-dim", type=int, default=13)
    parser.add_argument("--show-structure", action="store_true")
    args = parser.parse_args()

    metadata = load_metadata(args.metadata_json)
    field_dims = metadata["field_dims"]
    common_cfg = read_yaml(args.common_config)

    for model_name in args.models:
        cfg = merge_dict(common_cfg, {"model": model_name})
        model = _build_model(cfg, field_dims=field_dims, dense_dim=args.dense_dim)
        total, trainable = _count_params(model)
        print(f"\n=== {model_name} ===")
        print(f"total_params: {_format_num(total)}")
        print(f"trainable_params: {_format_num(trainable)}")
        print("params_by_top_module:")
        for k, v in _group_params(model).items():
            print(f"  - {k}: {_format_num(v)}")
        if args.show_structure:
            print("structure:")
            print(model)


if __name__ == "__main__":
    main()

