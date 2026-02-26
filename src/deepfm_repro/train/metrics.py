from __future__ import annotations

import numpy as np


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    order = np.argsort(y_prob)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_prob) + 1, dtype=np.float64)
    pos = y_true == 1
    n_pos = np.sum(pos)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = np.sum(ranks[pos])
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

