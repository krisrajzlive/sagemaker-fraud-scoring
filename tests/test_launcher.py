from __future__ import annotations

import os
import subprocess

import pytest

from fraud_scoring import launcher
from fraud_scoring.launcher import (
    LocalModeError,
    ensure_docker,
    latest_model_dir,
    stage_source_dir,
    to_sagemaker_hyperparameters,
)
from fraud_scoring.training import Hyperparameters


def test_staged_source_dir_is_self_contained(tmp_path):
    stage_source_dir(tmp_path)
    assert (tmp_path / "train.py").exists()
    assert (tmp_path / "fraud_scoring" / "entrypoints" / "train.py").exists()
    assert not list(tmp_path.rglob("__pycache__"))
    compile((tmp_path / "train.py").read_text(), "train.py", "exec")


def test_hyperparameters_are_strings_with_comma_joined_lists():
    values = to_sagemaker_hyperparameters(Hyperparameters(hidden_dims=(128, 64, 32), epochs=7))
    assert values["hidden_dims"] == "128,64,32"
    assert values["epochs"] == "7"
    assert all(isinstance(v, str) for v in values.values())
    assert "batch_size" in values  # snake_case names, as SageMaker conventions expect


def test_ensure_docker_explains_each_failure(monkeypatch):
    def fail_with(exc):
        def run(*_args, **_kwargs):
            raise exc

        return run

    cases = [
        (FileNotFoundError(), "not found on PATH"),
        (subprocess.CalledProcessError(1, "docker"), "not reachable"),
        (subprocess.TimeoutExpired("docker", 1), "timed out"),
    ]
    for exc, message in cases:
        monkeypatch.setattr(launcher.subprocess, "run", fail_with(exc))
        with pytest.raises(LocalModeError, match=message):
            ensure_docker()


def test_latest_model_dir_picks_the_most_recently_trained_job_regardless_of_name(tmp_path):
    # (job name, age in seconds, finished?) - the newest finished job sorts first by name, and
    # an unfinished job (no metadata.json) is newer still but must be ignored.
    jobs = (("a-newest", 10, True), ("m-middle", 500, True), ("z-oldest", 9000, True))
    jobs += (("b-unfinished", 1, False),)
    now = 2_000_000_000
    for job, age, finished in jobs:
        model = tmp_path / job / "model"
        model.mkdir(parents=True)
        if finished:
            metadata = model / "metadata.json"
            metadata.write_text("{}")
            os.utime(metadata, (now - age, now - age))
    assert latest_model_dir(tmp_path) == tmp_path / "a-newest" / "model"


def test_latest_model_dir_without_jobs_points_at_the_train_command(tmp_path):
    with pytest.raises(LocalModeError, match="fraud-scoring train"):
        latest_model_dir(tmp_path)


def test_launch_requires_prepared_data(tmp_path):
    with pytest.raises(LocalModeError, match="data prepare"):
        launcher.launch_local_training(
            data_dir=tmp_path / "missing",
            jobs_dir=tmp_path / "jobs",
            hyperparameters=Hyperparameters(),
            image="irrelevant",
        )
