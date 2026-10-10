"""SageMaker real-time inference handlers for the AWS PyTorch inference container.

The container (TorchServe plus the SageMaker inference toolkit) looks for four functions in
``code/inference.py``: ``model_fn``, ``input_fn``, ``predict_fn`` and ``output_fn``. They must
keep exactly these parameter lists because the toolkit counts them.

Request (``Content-Type: application/json``)::

    {"instances": [{"Time": 406.0, "V1": -2.31, ..., "V28": -0.02, "Amount": 149.62}, ...]}

Response::

    {"predictions": [{"fraud_probability": 0.9981, "flagged": true}, ...], "threshold": 0.9376}

Malformed requests get a 4xx status with a JSON ``{"error": ...}`` body. Anything unexpected is a
500. Only NumPy, pandas and PyTorch are imported here, because the inference container does not
ship the CLI's dependencies (typer, pydantic-settings, the SageMaker SDK).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_scoring.features import engineer
from fraud_scoring.model import ScoringModel, load_artifact
from fraud_scoring.schema import INPUT_FIELDS

JSON = "application/json"
MAX_INSTANCES = 1000

try:
    from sagemaker_inference.errors import BaseInferenceToolkitError
except ImportError:  # outside the AWS container (unit tests, local scripts)

    class BaseInferenceToolkitError(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int, message: str, phrase: str) -> None:
            self.status_code = status_code
            self.message = message
            self.phrase = phrase
            super().__init__(status_code, message, phrase)


class RequestError(BaseInferenceToolkitError):
    """A problem with the request itself; the toolkit turns ``status_code`` into the HTTP status."""

    def __init__(self, status_code: int, phrase: str, message: str) -> None:
        super().__init__(status_code, json.dumps({"error": message}), phrase)


def _bad_request(message: str) -> RequestError:
    return RequestError(400, "Bad Request", message)


def model_fn(model_dir: str) -> ScoringModel:
    return load_artifact(Path(model_dir))


def _media_type(header: str | None) -> str:
    return (header or "").split(";")[0].strip().lower()


def input_fn(request_body: str | bytes, content_type: str) -> np.ndarray:
    if _media_type(content_type) != JSON:
        raise RequestError(415, "Unsupported Media Type", f"send {JSON}, got {content_type!r}")
    try:
        payload = json.loads(request_body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise _bad_request("request body is not valid JSON") from exc

    instances: Any = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(instances, list) or not instances:
        raise _bad_request('expected {"instances": [<transaction>, ...]} with at least one item')
    if len(instances) > MAX_INSTANCES:
        raise RequestError(
            413, "Payload Too Large", f"at most {MAX_INSTANCES} instances per request"
        )
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict):
            raise _bad_request(f"instance {index} must be a JSON object")
        missing = [field for field in INPUT_FIELDS if field not in instance]
        if missing:
            raise _bad_request(f"instance {index} is missing fields: {', '.join(missing)}")

    try:
        return engineer(pd.DataFrame(instances))
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc


def predict_fn(features: np.ndarray, model: ScoringModel) -> dict[str, Any]:
    return {"probabilities": model.score(features), "threshold": model.threshold}


def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    if _media_type(accept) not in (JSON, "*/*", ""):
        raise RequestError(406, "Not Acceptable", f"this endpoint only returns {JSON}")
    threshold = prediction["threshold"]
    body = {
        "predictions": [
            {"fraud_probability": float(p), "flagged": bool(p >= threshold)}
            for p in prediction["probabilities"]
        ],
        "threshold": threshold,
    }
    return json.dumps(body), JSON
