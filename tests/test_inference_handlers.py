from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fraud_scoring.entrypoints.inference import (
    MAX_INSTANCES,
    RequestError,
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
)
from fraud_scoring.features import engineer
from fraud_scoring.model import load_artifact
from fraud_scoring.schema import INPUT_FIELDS
from tests.conftest import make_transactions


@pytest.fixture
def instances():
    return make_transactions(20, fraud_rate=0.3, seed=9)[INPUT_FIELDS].to_dict(orient="records")


def _request(instances) -> str:
    return json.dumps({"instances": instances})


def test_handlers_reproduce_the_scoring_model(trained_model, instances):
    model = model_fn(str(trained_model))
    features = input_fn(_request(instances), "application/json")
    body, content_type = output_fn(predict_fn(features, model), "application/json")

    assert content_type == "application/json"
    response = json.loads(body)
    expected = load_artifact(trained_model).score(engineer(pd.DataFrame(instances)))
    got = np.array([p["fraud_probability"] for p in response["predictions"]])
    assert np.array_equal(got, expected)
    assert response["threshold"] == model.threshold
    for prediction in response["predictions"]:
        assert prediction["flagged"] == (prediction["fraud_probability"] >= model.threshold)


def test_input_accepts_bytes_charset_parameters_and_ignores_extra_fields(instances):
    labelled = [{**row, "Class": 1, "note": "extra"} for row in instances]
    body = _request(labelled).encode()
    features = input_fn(body, "Application/JSON; charset=utf-8")
    assert features.shape[0] == len(instances)


@pytest.mark.parametrize(
    ("body", "content_type", "status", "fragment"),
    [
        ("{oops", "application/json", 400, "not valid JSON"),
        (b"\xff\xfe", "application/json", 400, "not valid JSON"),
        ("[1, 2]", "application/json", 400, "instances"),
        ('{"instances": []}', "application/json", 400, "at least one"),
        ('{"instances": "x"}', "application/json", 400, "instances"),
        ('{"instances": [1]}', "application/json", 400, "instance 0 must be a JSON object"),
        ('{"instances": [{"Time": 1}]}', "application/json", 400, "missing fields: V1"),
        ("{}", "text/csv", 415, "application/json"),
        ("{}", "", 415, "application/json"),
    ],
)
def test_bad_requests_get_4xx_with_a_json_error(body, content_type, status, fragment):
    with pytest.raises(RequestError) as caught:
        input_fn(body, content_type)
    assert caught.value.status_code == status
    assert fragment in json.loads(caught.value.message)["error"]


def test_value_errors_from_feature_engineering_become_400(instances):
    for bad, fragment in (({"Amount": -1.0}, "non-negative"), ({"V2": "abc"}, "non-numeric")):
        with pytest.raises(RequestError) as caught:
            input_fn(_request([{**instances[0], **bad}]), "application/json")
        assert caught.value.status_code == 400
        assert fragment in json.loads(caught.value.message)["error"]


def test_the_error_names_the_offending_instance(instances):
    broken = {k: v for k, v in instances[1].items() if k != "V9"}
    with pytest.raises(RequestError) as caught:
        input_fn(_request([instances[0], broken]), "application/json")
    assert "instance 1" in json.loads(caught.value.message)["error"]


def test_batch_size_is_capped(instances):
    too_many = [instances[0]] * (MAX_INSTANCES + 1)
    with pytest.raises(RequestError) as caught:
        input_fn(_request(too_many), "application/json")
    assert caught.value.status_code == 413
    assert input_fn(_request([instances[0]] * MAX_INSTANCES), "application/json").shape[0] == (
        MAX_INSTANCES
    )


@pytest.mark.parametrize("accept", ["application/json", "*/*", "", "application/json; q=0.9"])
def test_output_accepts_json_and_wildcards(accept):
    body, content_type = output_fn({"probabilities": np.array([0.2]), "threshold": 0.5}, accept)
    assert content_type == "application/json"
    assert json.loads(body)["predictions"] == [{"fraud_probability": 0.2, "flagged": False}]


def test_output_rejects_other_media_types():
    with pytest.raises(RequestError) as caught:
        output_fn({"probabilities": np.array([0.2]), "threshold": 0.5}, "text/html")
    assert caught.value.status_code == 406


def test_threshold_boundary_is_inclusive():
    body, _ = output_fn({"probabilities": np.array([0.5, 0.4999]), "threshold": 0.5}, "*/*")
    assert [p["flagged"] for p in json.loads(body)["predictions"]] == [True, False]


def test_handler_module_needs_only_what_the_inference_container_ships():
    """The AWS inference image ships no typer, rich, pydantic-settings or SageMaker SDK."""
    probe = (
        "import sys, fraud_scoring.entrypoints.inference;"
        "banned = {'sagemaker', 'typer', 'rich', 'pydantic_settings'};"
        "loaded = sorted(banned & {m.split('.')[0] for m in sys.modules});"
        "print(loaded); sys.exit(1 if loaded else 0)"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
