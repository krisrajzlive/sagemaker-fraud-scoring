from __future__ import annotations

import re

import numpy as np
import pytest

from fraud_scoring.features import engineer
from fraud_scoring.metrics import average_precision
from fraud_scoring.model import (
    CONFIG_FILE,
    METADATA_FILE,
    MODEL_FILE,
    PREPROCESSING_FILE,
    load_artifact,
    save_artifact,
)
from fraud_scoring.sagemaker_config import METRIC_DEFINITIONS
from fraud_scoring.schema import TARGET
from fraud_scoring.training import Hyperparameters, train
from tests.conftest import make_transactions

FAST = Hyperparameters(epochs=8, batch_size=256, hidden_dims=(16, 8), patience=3)


@pytest.fixture(scope="module")
def datasets():
    train_frame = make_transactions(4000, seed=1)
    validation_frame = make_transactions(1500, seed=2)
    return (
        engineer(train_frame),
        train_frame[TARGET].to_numpy(),
        engineer(validation_frame),
        validation_frame[TARGET].to_numpy(),
    )


@pytest.fixture(scope="module")
def log_lines() -> list[str]:
    return []


@pytest.fixture(scope="module")
def result(datasets, log_lines):
    return train(*datasets, FAST, log=log_lines.append)


def test_model_learns_the_planted_signal(result):
    assert result.validation["pr_auc"] > 0.8  # prevalence is ~0.05
    assert result.validation["roc_auc"] > 0.95
    assert 0.0 < result.threshold < 1.0
    assert 1 <= result.best_epoch <= FAST.epochs
    assert len(result.history) <= FAST.epochs


def test_training_is_deterministic_for_a_seed(datasets):
    first = train(*datasets, FAST, log=lambda _: None)
    second = train(*datasets, FAST, log=lambda _: None)
    assert [h["val_pr_auc"] for h in first.history] == [h["val_pr_auc"] for h in second.history]
    assert first.threshold == second.threshold


def test_early_stopping_halts_before_the_epoch_limit(datasets):
    impatient = Hyperparameters(epochs=60, batch_size=256, hidden_dims=(8,), patience=1, seed=3)
    outcome = train(*datasets, impatient, log=lambda _: None)
    assert len(outcome.history) < 60


def test_log_lines_match_the_sagemaker_metric_regexes(result, log_lines):
    epoch_lines = [line for line in log_lines if line.startswith("epoch=")]
    assert epoch_lines
    for definition in METRIC_DEFINITIONS:
        pattern = re.compile(definition["Regex"])
        values = [float(pattern.search(line).group(1)) for line in epoch_lines]
        assert len(values) == len(epoch_lines), definition["Name"]
    last = result.history[-1]
    assert float(re.search(METRIC_DEFINITIONS[1]["Regex"], epoch_lines[-1]).group(1)) == (
        pytest.approx(last["val_pr_auc"], abs=1e-5)
    )


def test_training_needs_both_classes(datasets):
    x_train, y_train, x_val, y_val = datasets
    with pytest.raises(ValueError, match="both classes"):
        train(x_train, np.zeros_like(y_train), x_val, y_val, FAST, log=lambda _: None)


def test_artifact_roundtrip_reproduces_scores(tmp_path, result, datasets):
    _, _, x_val, y_val = datasets
    metadata = {"threshold": result.threshold, "note": "unit test"}
    save_artifact(tmp_path, result.model, result.model_config, result.preprocessor, metadata)
    assert {p.name for p in tmp_path.iterdir()} == {
        MODEL_FILE,
        CONFIG_FILE,
        PREPROCESSING_FILE,
        METADATA_FILE,
    }

    loaded = load_artifact(tmp_path)
    scores = loaded.score(x_val)
    assert loaded.threshold == result.threshold
    assert average_precision(y_val, scores) == pytest.approx(result.validation["pr_auc"])


def test_artifact_loader_refuses_pickled_code(tmp_path, result, capsys):
    """model.pt is read with weights_only=True: a malicious pickle must neither load nor run."""
    import pickle

    save_artifact(
        tmp_path, result.model, result.model_config, result.preprocessor, {"threshold": 0.5}
    )

    class Payload:
        def __reduce__(self):
            return (print, ("payload-executed",))

    (tmp_path / MODEL_FILE).write_bytes(pickle.dumps(Payload()))
    capsys.readouterr()
    with pytest.raises(pickle.UnpicklingError):
        load_artifact(tmp_path)
    assert "payload-executed" not in capsys.readouterr().out
