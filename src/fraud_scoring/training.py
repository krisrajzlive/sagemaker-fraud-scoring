"""Training loop: weighted binary cross-entropy, early stopping on validation PR-AUC.

Plain functions over NumPy arrays so the same code runs inside the SageMaker training container
(via ``fraud_scoring.entrypoints.train``), on the host, and in the unit tests.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from fraud_scoring.features import Preprocessor
from fraud_scoring.metrics import average_precision, confusion_at, roc_auc, threshold_for_max_f1
from fraud_scoring.model import ModelConfig, RiskMLP, predict_proba


@dataclass(frozen=True)
class Hyperparameters:
    epochs: int = 40
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dims: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    # The positive class is weighted by (negatives / positives) ** pos_weight_power: 0 means
    # unweighted, 1 fully balanced. The square root is a common compromise for rare fraud.
    pos_weight_power: float = 0.5
    patience: int = 6
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingResult:
    model: RiskMLP
    model_config: ModelConfig
    preprocessor: Preprocessor
    threshold: float
    best_epoch: int
    history: list[dict[str, float]]
    validation: dict[str, float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    hyperparameters: Hyperparameters,
    log: Callable[[str], None] = print,
) -> TrainingResult:
    """Fit the network on engineered, unscaled features and return the best epoch's model.

    The preprocessor is fitted on ``x_train`` only. The decision threshold is chosen on the
    validation split (max F1) and is part of the result; the test split is never touched here.
    """
    hp = hyperparameters
    seed_everything(hp.seed)

    preprocessor = Preprocessor().fit(x_train)
    features = torch.from_numpy(preprocessor.transform(x_train))
    labels = torch.from_numpy(y_train.astype(np.float32))
    validation_features = preprocessor.transform(x_validation)

    positives = float(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("the training split needs both classes")
    pos_weight = ((len(labels) - positives) / positives) ** hp.pos_weight_power

    model_config = ModelConfig(
        input_dim=features.shape[1], hidden_dims=hp.hidden_dims, dropout=hp.dropout
    )
    model = RiskMLP(model_config)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay
    )
    shuffler = torch.Generator().manual_seed(hp.seed)

    log(
        f"training rows={len(labels)} positives={int(positives)} pos_weight={pos_weight:.2f} "
        f"features={features.shape[1]}"
    )
    history: list[dict[str, float]] = []
    best_score, best_epoch, best_state, stale = -1.0, 0, copy.deepcopy(model.state_dict()), 0

    for epoch in range(1, hp.epochs + 1):
        model.train()
        order = torch.randperm(len(labels), generator=shuffler)
        running = 0.0
        for start in range(0, len(order), hp.batch_size):
            batch = order[start : start + hp.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(features[batch]), labels[batch])
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)

        scores = predict_proba(model, validation_features)
        record = {
            "epoch": float(epoch),
            "train_loss": running / len(labels),
            "val_pr_auc": average_precision(y_validation, scores),
            "val_roc_auc": roc_auc(y_validation, scores),
        }
        history.append(record)
        # One key=value line per metric: the same format the SageMaker metric_definitions
        # regexes in fraud_scoring.sagemaker_config parse on a real training job.
        log(
            f"epoch={epoch} train_loss={record['train_loss']:.5f} "
            f"val_pr_auc={record['val_pr_auc']:.5f} val_roc_auc={record['val_roc_auc']:.5f}"
        )

        if record["val_pr_auc"] > best_score + 1e-6:
            best_score, best_epoch, stale = record["val_pr_auc"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= hp.patience:
                log(f"early stopping at epoch {epoch}; best epoch {best_epoch}")
                break

    model.load_state_dict(best_state)
    scores = predict_proba(model, validation_features)
    threshold = threshold_for_max_f1(y_validation, scores)
    validation = {
        "pr_auc": average_precision(y_validation, scores),
        "roc_auc": roc_auc(y_validation, scores),
        **confusion_at(y_validation, scores, threshold),
    }
    log(
        f"best_epoch={best_epoch} threshold={threshold:.4f} "
        f"validation_precision={validation['precision']:.4f} "
        f"validation_recall={validation['recall']:.4f} validation_f1={validation['f1']:.4f}"
    )
    return TrainingResult(
        model=model,
        model_config=model_config,
        preprocessor=preprocessor,
        threshold=threshold,
        best_epoch=best_epoch,
        history=history,
        validation=validation,
    )
