"""The scoring network and its on-disk artifact format.

An artifact is a directory (what SageMaker unpacks from ``model.tar.gz`` into ``/opt/ml/model``):

    model.pt            torch state_dict, loaded with ``weights_only=True`` (no pickled code)
    model_config.json   network architecture
    preprocessing.json  feature scaling fitted on the training split
    metadata.json       decision threshold, validation metrics, hyperparameters, lineage

Nothing in it depends on the NumPy or Python version that wrote it, which matters because the
training and inference containers ship different NumPy releases.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from fraud_scoring.features import Preprocessor
from fraud_scoring.schema import FEATURE_NAMES

MODEL_FILE = "model.pt"
CONFIG_FILE = "model_config.json"
PREPROCESSING_FILE = "preprocessing.json"
METADATA_FILE = "metadata.json"


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = len(FEATURE_NAMES)
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1


class RiskMLP(nn.Module):
    """Feed-forward network that outputs one fraud logit per transaction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = config.input_dim
        for hidden in config.hidden_dims:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(config.dropout)]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def predict_proba(model: nn.Module, scaled: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Fraud probabilities for already-scaled features."""
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(scaled), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(scaled[start : start + batch_size]))
            outputs.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(outputs).astype(np.float64) if outputs else np.empty(0)


@dataclass
class ScoringModel:
    """A loaded artifact: network, preprocessing and the decision threshold."""

    model: RiskMLP
    preprocessor: Preprocessor
    threshold: float
    metadata: dict[str, Any]

    def score(self, features: np.ndarray) -> np.ndarray:
        """Fraud probabilities for engineered, unscaled features (see ``features.engineer``)."""
        return predict_proba(self.model, self.preprocessor.transform(features))


def save_artifact(
    model_dir: Path,
    model: RiskMLP,
    config: ModelConfig,
    preprocessor: Preprocessor,
    metadata: dict[str, Any],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / MODEL_FILE)
    (model_dir / CONFIG_FILE).write_text(json.dumps(asdict(config), indent=2))
    (model_dir / PREPROCESSING_FILE).write_text(json.dumps(preprocessor.to_dict()))
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def load_artifact(model_dir: Path) -> ScoringModel:
    config_raw = json.loads((model_dir / CONFIG_FILE).read_text())
    config = ModelConfig(
        input_dim=config_raw["input_dim"],
        hidden_dims=tuple(config_raw["hidden_dims"]),
        dropout=config_raw["dropout"],
    )
    model = RiskMLP(config)
    state = torch.load(model_dir / MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    preprocessor = Preprocessor.from_dict(json.loads((model_dir / PREPROCESSING_FILE).read_text()))
    metadata = json.loads((model_dir / METADATA_FILE).read_text())
    return ScoringModel(model, preprocessor, float(metadata["threshold"]), metadata)
