"""Held-out evaluation of a trained artifact, with uncertainty and baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fraud_scoring.data import read_split
from fraud_scoring.features import engineer
from fraud_scoring.metrics import average_precision, bootstrap_ci, confusion_at, roc_auc
from fraud_scoring.model import load_artifact
from fraud_scoring.schema import TARGET


def _ranking(y: Any, scores: Any, n_bootstrap: int, seed: int) -> dict[str, Any]:
    return {
        "pr_auc": average_precision(y, scores),
        "pr_auc_ci95": list(
            bootstrap_ci(average_precision, y, scores, n_resamples=n_bootstrap, seed=seed)
        ),
        "roc_auc": roc_auc(y, scores),
        "roc_auc_ci95": list(bootstrap_ci(roc_auc, y, scores, n_resamples=n_bootstrap, seed=seed)),
    }


def evaluate_artifact(
    model_dir: Path,
    test_dir: Path,
    *,
    train_dir: Path | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score the test split with the artifact's own threshold and compare against baselines.

    The threshold comes from the artifact (chosen on the validation split), so nothing here is
    tuned on the test data. Baselines: the positive-class prevalence (what random scores give)
    and a class-balanced logistic regression fitted on the training split with the same features.
    """
    scoring = load_artifact(model_dir)
    test = read_split(test_dir)
    y = test[TARGET].to_numpy()
    features = engineer(test)
    scores = scoring.score(features)

    report: dict[str, Any] = {
        "test_rows": int(len(test)),
        "test_frauds": int(y.sum()),
        "model": {
            **_ranking(y, scores, n_bootstrap, seed),
            "at_threshold": confusion_at(y, scores, scoring.threshold),
            "threshold_rule": scoring.metadata.get("threshold_rule"),
        },
        "baselines": {"prevalence": {"pr_auc": float(y.mean())}},
    }

    if train_dir is not None:
        from sklearn.linear_model import LogisticRegression

        train = read_split(train_dir)
        baseline = LogisticRegression(class_weight="balanced", max_iter=1000)
        baseline.fit(scoring.preprocessor.transform(engineer(train)), train[TARGET].to_numpy())
        baseline_scores = baseline.predict_proba(scoring.preprocessor.transform(features))[:, 1]
        report["baselines"]["logistic_regression"] = _ranking(y, baseline_scores, n_bootstrap, seed)
    return report
