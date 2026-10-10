from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from fraud_scoring.cli import app
from fraud_scoring.data import split_by_time, write_splits

runner = CliRunner()


def _words(output: str) -> str:
    return " ".join(output.split())


def test_predict_file_shows_scores_and_decisions(endpoint_url, tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps({"instances": [{"Amount": 5}, {"Amount": 500}, {"Amount": 20}]}))
    result = runner.invoke(app, ["predict", "--url", endpoint_url, "--file", str(requests)])
    assert result.exit_code == 0, result.output
    text = _words(result.output)
    assert "FLAGGED" in text and "500.00" in text and "0.9900" in text
    assert "1 of 3 flagged" in text
    assert "actual" not in text, "no labels exist for hand-written requests"


def test_predict_accepts_a_bare_list(endpoint_url, tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"Amount": 150}]))
    result = runner.invoke(app, ["predict", "--url", endpoint_url, "--file", str(requests)])
    assert result.exit_code == 0 and "1 of 1 flagged" in _words(result.output)


def test_predict_sample_compares_with_the_true_labels(endpoint_url, transactions):
    write_splits(split_by_time(transactions), Path("data/processed"), "synthetic")
    result = runner.invoke(
        app, ["predict", "--url", endpoint_url, "--sample", "6", "--frauds", "2"]
    )
    assert result.exit_code == 0, result.output
    text = _words(result.output)
    assert "actual" in text and "FRAUD" in text and "legit" in text
    assert "decisions match the true label" in text


def test_predict_reports_an_unreachable_endpoint_cleanly(tmp_path):
    requests = tmp_path / "requests.json"
    requests.write_text(json.dumps([{"Amount": 1}]))
    result = runner.invoke(app, ["predict", "--url", "http://127.0.0.1:9", "--file", str(requests)])
    assert result.exit_code == 1
    assert "cannot reach" in _words(result.output)


def test_predict_without_prepared_data_explains_what_to_do():
    result = runner.invoke(app, ["predict", "--url", "http://127.0.0.1:9"])
    assert result.exit_code == 1 and "data prepare" in _words(result.output)


@pytest.mark.parametrize("command", [["predict", "--sample", "2"], ["serve"]])
def test_commands_that_start_an_endpoint_need_a_trained_model(command, transactions):
    write_splits(split_by_time(transactions), Path("data/processed"), "synthetic")
    result = runner.invoke(app, command)
    assert result.exit_code == 1 and "fraud-scoring train" in _words(result.output)


def test_help_lists_the_inference_commands():
    output = runner.invoke(app, ["--help"]).output
    assert "predict" in output and "serve" in output
