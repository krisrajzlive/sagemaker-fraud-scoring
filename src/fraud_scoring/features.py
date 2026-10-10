"""Feature engineering and scaling, shared by the training container and the scoring endpoint."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_scoring.schema import FEATURE_NAMES, INPUT_FIELDS, V_COLUMNS

SECONDS_PER_DAY = 86_400
PREPROCESSING_VERSION = 1


def engineer(frame: pd.DataFrame) -> np.ndarray:
    """Turn raw transaction fields into unscaled model features (float32, one row per row).

    ``V1..V28`` pass through, ``Amount`` becomes ``log1p(Amount)`` and ``Time`` (seconds since the
    first transaction) becomes a sine/cosine pair of the time of day. Raises ``ValueError`` on
    missing fields, non-numeric or non-finite values and negative amounts, so a bad scoring
    request fails loudly instead of producing a plausible-looking score.
    """
    missing = [name for name in INPUT_FIELDS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing fields: {', '.join(missing)}")
    try:
        values = frame[INPUT_FIELDS].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric value in transaction fields: {exc}") from exc
    if not np.isfinite(values).all():
        raise ValueError("non-finite value (NaN or infinity) in transaction fields")

    time, amount = values[:, 0], values[:, -1]
    if (amount < 0).any():
        raise ValueError("Amount must be non-negative")
    angle = 2.0 * np.pi * np.mod(time, SECONDS_PER_DAY) / SECONDS_PER_DAY
    v = values[:, 1 : 1 + len(V_COLUMNS)]
    features = np.column_stack([v, np.log1p(amount), np.sin(angle), np.cos(angle)])
    return features.astype(np.float32)


@dataclass
class Preprocessor:
    """Per-feature standardisation, fitted on the training split only and shipped with the model."""

    mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    scale: np.ndarray = field(default_factory=lambda: np.ones(0))

    def fit(self, features: np.ndarray) -> Preprocessor:
        std = features.std(axis=0, dtype=np.float64)
        self.mean = features.mean(axis=0, dtype=np.float64)
        self.scale = np.where(std > 1e-8, std, 1.0)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError(
                f"expected a (n, {len(self.mean)}) feature matrix, got shape {features.shape}"
            )
        return ((features - self.mean) / self.scale).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PREPROCESSING_VERSION,
            "feature_names": FEATURE_NAMES,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Preprocessor:
        if payload.get("version") != PREPROCESSING_VERSION:
            raise ValueError(f"unsupported preprocessing version: {payload.get('version')!r}")
        if payload.get("feature_names") != FEATURE_NAMES:
            raise ValueError("preprocessing feature names do not match this code version")
        return cls(mean=np.asarray(payload["mean"]), scale=np.asarray(payload["scale"]))
