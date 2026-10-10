from __future__ import annotations

import json

from typer.testing import CliRunner

from fraud_scoring.cli import app
from fraud_scoring.config import get_settings
from fraud_scoring.evaluation import evaluate_artifact

runner = CliRunner()


def test_evaluation_reports_model_against_baselines(trained_model, processed_dir):
    report = evaluate_artifact(
        trained_model,
        processed_dir / "test",
        train_dir=processed_dir / "train",
        n_bootstrap=100,
    )
    prevalence = report["baselines"]["prevalence"]["pr_auc"]
    assert report["model"]["pr_auc"] > 5 * prevalence
    low, high = report["model"]["pr_auc_ci95"]
    assert low <= report["model"]["pr_auc"] <= high
    assert "logistic_regression" in report["baselines"]
    at = report["model"]["at_threshold"]
    assert at["tp"] + at["fn"] == report["test_frauds"]
    assert at["tp"] + at["fp"] + at["fn"] + at["tn"] == report["test_rows"]
    json.dumps(report)  # must be serialisable as-is


def test_evaluation_without_train_split_skips_the_learned_baseline(trained_model, processed_dir):
    report = evaluate_artifact(trained_model, processed_dir / "test", n_bootstrap=20)
    assert set(report["baselines"]) == {"prevalence"}


def test_cli_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("data", "train", "evaluate"):
        assert command in result.output


def test_data_prepare_without_download_fails_with_guidance():
    result = runner.invoke(app, ["data", "prepare"])
    assert result.exit_code == 1
    assert "data download" in " ".join(result.output.split())


def test_data_prepare_reports_a_corrupted_download_without_a_traceback():
    raw = get_settings().raw_dir
    raw.mkdir(parents=True)
    (raw / "creditcard.csv").write_text("not the dataset")
    result = runner.invoke(app, ["data", "prepare"])
    assert result.exit_code == 1
    assert "sha256" in result.output
    assert isinstance(result.exception, SystemExit)  # a clean exit, not a traceback


def test_evaluate_without_a_trained_model_fails_with_guidance():
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 1
    assert "fraud-scoring train" in result.output
