"""Training in the real AWS PyTorch container. Needs Docker; run with ``pytest -m docker``."""

from __future__ import annotations

import subprocess
import tarfile

import pytest

from fraud_scoring.config import DEFAULT_TRAINING_IMAGE
from fraud_scoring.launcher import launch_local_training
from fraud_scoring.model import load_artifact
from fraud_scoring.training import Hyperparameters

pytestmark = pytest.mark.docker


def test_training_job_runs_in_the_sagemaker_container(processed_dir, tmp_path):
    run = launch_local_training(
        data_dir=processed_dir,
        jobs_dir=tmp_path / "jobs",
        hyperparameters=Hyperparameters(epochs=2, hidden_dims=(8,), batch_size=512),
        image=DEFAULT_TRAINING_IMAGE,
        job_name="integration-test",
    )

    with tarfile.open(run.model_tar) as archive:
        names = {member.name.removeprefix("./") for member in archive.getmembers()}
    assert {"model.pt", "model_config.json", "preprocessing.json", "metadata.json"} <= names

    scoring = load_artifact(run.model_dir)
    runtime = scoring.metadata["runtime"]
    assert runtime["python"].startswith("3.11"), "must have run in the container, not on the host"
    assert runtime["torch"].startswith("2.5.1")
    assert scoring.metadata["data_manifest"]["dataset_sha256"] == "synthetic"


def test_job_leaves_no_container_behind(processed_dir, tmp_path):
    launch_local_training(
        data_dir=processed_dir,
        jobs_dir=tmp_path / "jobs",
        hyperparameters=Hyperparameters(epochs=1, hidden_dims=(4,)),
        image=DEFAULT_TRAINING_IMAGE,
        job_name="cleanup-check",
    )
    listing = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=cleanup-check", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert listing.stdout.strip() == ""
