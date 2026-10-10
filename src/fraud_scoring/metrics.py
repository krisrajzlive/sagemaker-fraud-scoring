"""Ranking and threshold metrics for a rare-positive problem, in plain NumPy.

Kept free of scikit-learn and SciPy so the same code runs in the training container, the
inference container and the host. The tests check every function against scikit-learn.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cumulative false/true positives at each distinct score, highest score first."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and scores must be 1-D arrays of the same length")
    if not y.any():
        raise ValueError("y_true has no positive examples")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last_of_run = np.r_[np.flatnonzero(np.diff(s)), y.size - 1]
    tps = np.cumsum(y)[last_of_run]
    fps = (1 + last_of_run) - tps
    return fps, tps, s[last_of_run]


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve as a step sum (scikit-learn's definition)."""
    fps, tps, _ = _curve(y_true, scores)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (trapezoid rule over distinct thresholds)."""
    fps, tps, _ = _curve(y_true, scores)
    if fps[-1] == 0:
        raise ValueError("y_true has no negative examples")
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def confusion_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Counts and precision/recall/F1 when predicting positive for ``score >= threshold``."""
    y = np.asarray(y_true).astype(bool)
    predicted = np.asarray(scores, dtype=np.float64) >= threshold
    tp = int(np.sum(predicted & y))
    fp = int(np.sum(predicted & ~y))
    fn = int(np.sum(~predicted & y))
    tn = int(np.sum(~predicted & ~y))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def threshold_for_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The score cut-off (predict positive when ``score >= cut-off``) that maximises F1."""
    fps, tps, thresholds = _curve(y_true, scores)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling positives and negatives separately.

    With only a few dozen positives in a held-out split, a single metric value is noisy; the
    interval shows how much.
    """
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    positives, negatives = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    for i in range(n_resamples):
        index = np.concatenate(
            [
                rng.choice(positives, size=positives.size, replace=True),
                rng.choice(negatives, size=negatives.size, replace=True),
            ]
        )
        values[i] = metric(y[index], s[index])
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(values, [tail, 100.0 - tail])
    return float(low), float(high)
