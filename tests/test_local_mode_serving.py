"""Serving in the real AWS PyTorch inference container. Needs Docker: ``pytest -m docker``."""

from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from fraud_scoring.client import EndpointError, invoke
from fraud_scoring.config import DEFAULT_INFERENCE_IMAGE
from fraud_scoring.features import engineer
from fraud_scoring.model import load_artifact
from fraud_scoring.schema import INPUT_FIELDS
from fraud_scoring.serving import LocalEndpoint
from tests.conftest import make_transactions

pytestmark = pytest.mark.docker

TEST_PORT = 8099  # away from 8080, which other local services commonly use


def _inference_containers() -> set[str]:
    listing = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "-q",
            "--no-trunc",
            "--filter",
            f"ancestor={DEFAULT_INFERENCE_IMAGE}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(listing.stdout.split())


def test_endpoint_serves_the_model_and_cleans_up_after_itself(trained_model, tmp_path):
    before = _inference_containers()
    instances = make_transactions(30, fraud_rate=0.3, seed=4)[INPUT_FIELDS].to_dict("records")

    with LocalEndpoint(
        trained_model, DEFAULT_INFERENCE_IMAGE, port=TEST_PORT, log_path=tmp_path / "endpoint.log"
    ) as endpoint:
        response = invoke(endpoint.url, instances)

        host = load_artifact(trained_model).score(engineer(pd.DataFrame(instances)))
        served = np.array([p["fraud_probability"] for p in response["predictions"]])
        np.testing.assert_allclose(served, host, rtol=0, atol=1e-6)

        with pytest.raises(EndpointError) as caught:
            invoke(endpoint.url, [{"Amount": 1.0}])
        assert caught.value.status == 400 and "missing fields" in caught.value.detail

    assert _inference_containers() == before, "the endpoint leaked a container"
    assert not endpoint.alive()


def test_starting_on_a_busy_port_fails_fast_without_side_effects(trained_model, tmp_path):
    import socket

    from fraud_scoring.launcher import LocalModeError

    before = _inference_containers()
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        endpoint = LocalEndpoint(
            trained_model,
            DEFAULT_INFERENCE_IMAGE,
            port=busy.getsockname()[1],
            log_path=tmp_path / "endpoint.log",
        )
        with pytest.raises(LocalModeError, match="already in use"):
            endpoint.start()
    assert _inference_containers() == before
