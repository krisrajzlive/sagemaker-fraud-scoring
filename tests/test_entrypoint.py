from __future__ import annotations

import json
import tarfile

import pytest

from fraud_scoring.entrypoints.train import hyperparameters_from, main, parse_args
from fraud_scoring.model import METADATA_FILE, load_artifact


def _run(processed_dir, tmp_path, *extra):
    model_dir, output_dir = tmp_path / "model", tmp_path / "output"
    main(
        [
            "--train", str(processed_dir / "train"),
            "--validation", str(processed_dir / "validation"),
            "--model-dir", str(model_dir),
            "--output-dir", str(output_dir),
            "--epochs", "3",
            "--hidden-dims", "8,4",
            *extra,
        ]
    )  # fmt: skip
    return model_dir, output_dir


def test_entrypoint_writes_a_loadable_artifact_with_lineage(processed_dir, tmp_path):
    model_dir, output_dir = _run(processed_dir, tmp_path)

    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    assert metadata["data_manifest"]["dataset_sha256"] == "synthetic"
    assert metadata["hyperparameters"]["hidden_dims"] == [8, 4]
    assert metadata["threshold_rule"] == "max F1 on the validation split"
    assert {"python", "torch", "numpy"} <= set(metadata["runtime"])
    assert load_artifact(model_dir).threshold == metadata["threshold"]

    history = json.loads((output_dir / "training_history.json").read_text())
    assert 1 <= len(history) <= 3


def test_underscore_hyperparameter_names_are_accepted(monkeypatch, processed_dir):
    """SageMaker passes hyperparameters as given, and the convention is snake_case."""
    args = parse_args(
        [
            "--train", str(processed_dir / "train"),
            "--validation", str(processed_dir / "validation"),
            "--batch_size", "128",
            "--learning_rate", "0.01",
            "--hidden_dims", "32,16,8",
            "--pos_weight_power", "1.0",
        ]
    )  # fmt: skip
    hp = hyperparameters_from(args)
    assert (hp.batch_size, hp.learning_rate, hp.hidden_dims, hp.pos_weight_power) == (
        128,
        0.01,
        (32, 16, 8),
        1.0,
    )


def test_channels_default_to_sagemaker_environment_variables(monkeypatch, processed_dir):
    monkeypatch.setenv("SM_CHANNEL_TRAIN", str(processed_dir / "train"))
    monkeypatch.setenv("SM_CHANNEL_VALIDATION", str(processed_dir / "validation"))
    monkeypatch.setenv("SM_MODEL_DIR", "/somewhere/model")
    args = parse_args([])
    assert args.train.endswith("train") and args.model_dir == "/somewhere/model"


def test_missing_channel_is_a_usage_error(monkeypatch):
    monkeypatch.delenv("SM_CHANNEL_TRAIN", raising=False)
    monkeypatch.delenv("SM_CHANNEL_VALIDATION", raising=False)
    with pytest.raises(SystemExit):
        parse_args([])


def test_artifact_survives_the_model_tar_gz_contract(processed_dir, tmp_path):
    """SageMaker ships /opt/ml/model as model.tar.gz and unpacks it for the inference container."""
    model_dir, _ = _run(processed_dir, tmp_path)
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in model_dir.iterdir():
            tar.add(path, arcname=path.name)
    unpacked = tmp_path / "unpacked"
    with tarfile.open(archive) as tar:
        tar.extractall(unpacked, filter="data")
    assert load_artifact(unpacked).threshold == load_artifact(model_dir).threshold
