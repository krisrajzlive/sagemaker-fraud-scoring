from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_scoring.metrics import (
    average_precision,
    bootstrap_ci,
    confusion_at,
    roc_auc,
    threshold_for_max_f1,
)


def _random_problem(seed: int, n: int = 500, tie_levels: int | None = None):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.1).astype(int)
    scores = rng.normal(size=n) + 1.5 * y
    if tie_levels:
        scores = np.round(scores * tie_levels) / tie_levels
    return y, scores


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("tie_levels", [None, 3])
def test_ranking_metrics_match_scikit_learn(seed, tie_levels):
    y, scores = _random_problem(seed, tie_levels=tie_levels)
    assert average_precision(y, scores) == pytest.approx(average_precision_score(y, scores))
    assert roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_perfect_and_reversed_ranking():
    y = np.array([0, 0, 1, 1])
    assert average_precision(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_metrics_reject_degenerate_labels():
    with pytest.raises(ValueError, match="no positive"):
        average_precision(np.zeros(5), np.arange(5.0))
    with pytest.raises(ValueError, match="no negative"):
        roc_auc(np.ones(5), np.arange(5.0))
    with pytest.raises(ValueError, match="same length"):
        average_precision(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))


def test_confusion_at_counts_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.2, 0.1, 0.05])
    result = confusion_at(y, scores, 0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 4)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["f1"] == pytest.approx(2 / 3)
    assert confusion_at(y, scores, 2.0)["precision"] == 0.0


@pytest.mark.parametrize("seed", range(3))
def test_threshold_for_max_f1_beats_every_other_candidate(seed):
    y, scores = _random_problem(seed, n=200)
    best = confusion_at(y, scores, threshold_for_max_f1(y, scores))["f1"]
    for candidate in np.unique(scores):
        assert confusion_at(y, scores, candidate)["f1"] <= best + 1e-12


def test_bootstrap_interval_brackets_estimate_and_is_reproducible():
    y, scores = _random_problem(1, n=400)
    low, high = bootstrap_ci(average_precision, y, scores, n_resamples=200, seed=7)
    assert low < average_precision(y, scores) < high
    assert (low, high) == bootstrap_ci(average_precision, y, scores, n_resamples=200, seed=7)
