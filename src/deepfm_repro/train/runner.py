from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfm_repro.data.dataset import CTRDataset, load_metadata, load_processed_data
from deepfm_repro.models import DeepFMModel, DNNModel, FMModel, FNNModel, LRModel, WideDeepModel
from deepfm_repro.train.metrics import binary_auc, binary_logloss
from deepfm_repro.utils import config_hash, set_seed


def _format_num(n: int) -> str:
    return f"{n:,}"


def _print_model_summary(model: nn.Module, model_name: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    module_counts: Dict[str, int] = {}
    for name, p in model.named_parameters():
        key = name.split(".")[0]
        module_counts[key] = module_counts.get(key, 0) + p.numel()

    print("\n===== Model Summary =====")
    print(f"model: {model_name}")
    print(f"total_params: {_format_num(total)}")
    print(f"trainable_params: {_format_num(trainable)}")
    print("params_by_top_module:")
    for key in sorted(module_counts.keys()):
        print(f"  - {key}: {_format_num(module_counts[key])}")
    print("structure:")
    print(model)
    print("=========================\n")


def _print_detailed_summary_with_torchinfo(
    model: nn.Module,
    model_name: str,
    dense_dim: int,
    sparse_fields: int,
    device: torch.device,
) -> None:
    try:
        from torchinfo import summary  # type: ignore
    except Exception:
        _print_model_summary(model, model_name)
        print("torchinfo not installed, fallback to simple summary. Install with: pip install torchinfo")
        return

    dense_sample = torch.zeros((2, dense_dim), dtype=torch.float32, device=device)
    sparse_sample = torch.zeros((2, sparse_fields), dtype=torch.long, device=device)
    try:
        print("\n===== Detailed Model Summary (torchinfo) =====")
        print(f"model: {model_name}")
        info = summary(
            model,
            input_data=(dense_sample, sparse_sample),
            depth=6,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=0,
            device=str(device),
        )
        print(info)
        print("==============================================\n")
    except Exception as exc:
        print(f"torchinfo summary failed: {exc}. Falling back to simple summary.")
        _print_model_summary(model, model_name)


def _build_model(config: Dict, field_dims: Iterable[int], dense_dim: int) -> nn.Module:
    model_name = config["model"]
    embedding_dim = int(config.get("embedding_dim", 16))
    mlp_dims = config.get("mlp_dims", [400, 400, 400])
    dropout = float(config.get("dropout", 0.2))
    field_dims = list(field_dims)
    if model_name == "lr":
        return LRModel(field_dims=field_dims, dense_dim=dense_dim)
    if model_name == "fm":
        return FMModel(field_dims=field_dims, dense_dim=dense_dim, embedding_dim=embedding_dim)
    if model_name == "fnn":
        return FNNModel(field_dims=field_dims, dense_dim=dense_dim, embedding_dim=embedding_dim, mlp_dims=mlp_dims, dropout=dropout)
    if model_name == "dnn":
        return DNNModel(field_dims=field_dims, dense_dim=dense_dim, embedding_dim=embedding_dim, mlp_dims=mlp_dims, dropout=dropout)
    if model_name == "wide_deep":
        return WideDeepModel(
            field_dims=field_dims,
            dense_dim=dense_dim,
            embedding_dim=embedding_dim,
            mlp_dims=mlp_dims,
            dropout=dropout,
        )
    if model_name == "deepfm":
        return DeepFMModel(field_dims=field_dims, dense_dim=dense_dim, embedding_dim=embedding_dim, mlp_dims=mlp_dims, dropout=dropout)
    raise ValueError(f"Unsupported model: {model_name}")


def _get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    losses = []
    labels = []
    probs = []
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for dense, sparse, y in loader:
            dense = dense.to(device)
            sparse = sparse.to(device)
            y = y.to(device).view(-1, 1)
            logits = model(dense, sparse)
            loss = criterion(logits, y)
            losses.append(float(loss.item()))
            prob = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            probs.append(prob)
            labels.append(y.cpu().numpy().reshape(-1))
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    return y_true, y_prob, float(np.mean(losses))


def _append_metrics(metrics_csv: Path, row: Dict[str, object]) -> None:
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = metrics_csv.exists()
    with metrics_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "model",
                "seed",
                "split",
                "auc",
                "logloss",
                "best_epoch",
                "config_hash",
                "timestamp",
                "init_from",
                "pretrain_run_id",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_once(config: Dict) -> Dict[str, object]:
    seed = int(config.get("seed", 2026))
    set_seed(seed)

    data_npz = config["data_npz"]
    metadata_json = config["metadata_json"]
    dataset = load_processed_data(data_npz)
    metadata = load_metadata(metadata_json)
    field_dims = metadata["field_dims"]

    train_set = CTRDataset(*dataset["train"])
    valid_set = CTRDataset(*dataset["valid"])
    test_set = CTRDataset(*dataset["test"])
    dense_dim = int(train_set.dense.shape[1])

    batch_size = int(config.get("batch_size", 4096))
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = _get_device(config.get("device", "cuda"))
    model = _build_model(config, field_dims, dense_dim).to(device)
    _print_detailed_summary_with_torchinfo(
        model=model,
        model_name=config["model"],
        dense_dim=dense_dim,
        sparse_fields=len(field_dims),
        device=device,
    )

    pretrain_ckpt = config.get("pretrain_checkpoint")
    if config["model"] == "fnn" and pretrain_ckpt:
        fm_ckpt = torch.load(pretrain_ckpt, map_location="cpu")
        model.init_from_fm_state(fm_ckpt["model_state_dict"])

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-6)),
    )
    criterion = nn.BCEWithLogitsLoss()
    epochs = int(config.get("epochs", 5))
    patience = int(config.get("early_stop_patience", 2))

    run_id = f"{config['model']}_{seed}_{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    results_dir = Path(config.get("results_dir", "results"))
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_epoch = -1
    stale = 0
    best_checkpoint = run_dir / "best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for dense, sparse, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True):
            dense = dense.to(device)
            sparse = sparse.to(device)
            y = y.to(device).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(dense, sparse)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_y, val_prob, _ = _predict(model, valid_loader, device)
        val_auc = binary_auc(val_y, val_prob)
        avg_train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        tqdm.write(f"[Epoch {epoch}/{epochs}] train_loss={avg_train_loss:.6f} valid_auc={val_auc:.6f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "run_id": run_id,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_valid_auc": best_auc,
                },
                best_checkpoint,
            )
        else:
            stale += 1
        if stale >= patience:
            break

    best = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    valid_y, valid_prob, _ = _predict(model, valid_loader, device)
    test_y, test_prob, _ = _predict(model, test_loader, device)

    valid_auc = binary_auc(valid_y, valid_prob)
    valid_logloss = binary_logloss(valid_y, valid_prob)
    test_auc = binary_auc(test_y, test_prob)
    test_logloss = binary_logloss(test_y, test_prob)

    metrics_csv = results_dir / "metrics.csv"
    cfg_hash = config_hash(config)
    init_from = config.get("init_from", "none")
    pretrain_run_id = config.get("pretrain_run_id", "")

    for split, auc, logloss in (
        ("valid", valid_auc, valid_logloss),
        ("test", test_auc, test_logloss),
    ):
        _append_metrics(
            metrics_csv,
            {
                "run_id": run_id,
                "model": config["model"],
                "seed": seed,
                "split": split,
                "auc": auc,
                "logloss": logloss,
                "best_epoch": best_epoch,
                "config_hash": cfg_hash,
                "timestamp": timestamp,
                "init_from": init_from,
                "pretrain_run_id": pretrain_run_id,
            },
        )

    summary = {
        "run_id": run_id,
        "model": config["model"],
        "seed": seed,
        "best_epoch": best_epoch,
        "valid": {"auc": valid_auc, "logloss": valid_logloss},
        "test": {"auc": test_auc, "logloss": test_logloss},
        "init_from": init_from,
        "pretrain_run_id": pretrain_run_id,
        "checkpoint": str(best_checkpoint),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def evaluate_run(run_dir: str | Path, data_npz: str | Path, metadata_json: str | Path, batch_size: int = 8192) -> Dict[str, float]:
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu")
    config = checkpoint["config"]
    metadata = load_metadata(metadata_json)
    field_dims = metadata["field_dims"]
    dataset = load_processed_data(data_npz)
    test_set = CTRDataset(*dataset["test"])
    dense_dim = int(test_set.dense.shape[1])
    model = _build_model(config, field_dims, dense_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = _get_device(config.get("device", "cpu"))
    model = model.to(device)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    y_true, y_prob, _ = _predict(model, test_loader, device)
    return {"auc": binary_auc(y_true, y_prob), "logloss": binary_logloss(y_true, y_prob)}
