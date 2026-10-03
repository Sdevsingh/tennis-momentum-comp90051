"""Evaluation metrics from scratch (brief Section 2.g).

Third-party metric functions are NOT allowed. These take predicted
probabilities p (for the positive class) and binary labels y.

Error bars across CV repetitions are the whole point of running >=10
repetitions, so `summarise` returns mean and a standard error.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    """Mean binary cross-entropy. Lower is better."""
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of probabilities. Lower is better."""
    return float(np.mean((p - y) ** 2))


def accuracy(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((p >= threshold).astype(int) == y))


def auc_roc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC via the rank/Mann-Whitney-U identity, computed by hand.

    AUC = P(score of a random positive > score of a random negative). Ties
    count as half. This averages pairwise comparisons in O(n log n) using
    rank sums rather than an O(n^2) double loop.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    # Average ranks within tie groups.
    sorted_p = p[order]
    i = 0
    rank = 1
    while i < len(sorted_p):
        j = i
        while j < len(sorted_p) and sorted_p[j] == sorted_p[i]:
            j += 1
        avg = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[order[i:j]] = avg
        rank += (j - i)
        i = j
    sum_ranks_pos = np.sum(ranks[y == 1])
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Reliability-curve data: mean predicted vs observed frequency per bin.

    Returns (mean_pred, obs_freq, counts). Feed to a plotting lib for a
    calibration plot — computing it here keeps us within the from-scratch rule.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    obs_freq = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(np.sum(m))
        if counts[b] > 0:
            mean_pred[b] = float(np.mean(p[m]))
            obs_freq[b] = float(np.mean(y[m]))
    return mean_pred, obs_freq, counts


def summarise(values: list[float]) -> dict[str, float]:
    """Mean and standard error across CV repetitions (for error bars)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    mean = float(np.mean(arr)) if n else float("nan")
    se = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"mean": mean, "se": se, "n": n}


ALL_METRICS = {
    "log_loss": log_loss,
    "brier": brier_score,
    "accuracy": accuracy,
    "auc": auc_roc,
}
