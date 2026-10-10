"""SageMaker script-mode training entrypoint.

Runs inside the AWS PyTorch training container. SageMaker passes hyperparameters as ``--name
value`` arguments and describes the job through ``SM_*`` environment variables: input channels
under ``SM_CHANNEL_<NAME>``, the model output directory in ``SM_MODEL_DIR`` (packed into
``model.tar.gz``) and extra output files in ``SM_OUTPUT_DATA_DIR`` (packed into ``output.tar.gz``).
Every path can also be given as an argument, so the script runs unchanged on a laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from fraud_scoring.data import read_manifest, read_split
from fraud_scoring.features import engineer
from fraud_scoring.model import save_artifact
from fraud_scoring.schema import TARGET
from fraud_scoring.training import Hyperparameters, train

DEFAULTS = Hyperparameters()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # SageMaker hyperparameters are conventionally snake_case, people type hyphens: accept both.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULTS.epochs)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=DEFAULTS.batch_size
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=DEFAULTS.learning_rate,
    )
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=DEFAULTS.weight_decay,
    )
    parser.add_argument(
        "--hidden-dims",
        "--hidden_dims",
        dest="hidden_dims",
        default=",".join(map(str, DEFAULTS.hidden_dims)),
    )
    parser.add_argument("--dropout", type=float, default=DEFAULTS.dropout)
    parser.add_argument(
        "--pos-weight-power",
        "--pos_weight_power",
        dest="pos_weight_power",
        type=float,
        default=DEFAULTS.pos_weight_power,
    )
    parser.add_argument("--patience", type=int, default=DEFAULTS.patience)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)

    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument(
        "--model-dir",
        "--model_dir",
        dest="model_dir",
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    args = parser.parse_args(argv)
    for name in ("train", "validation"):
        if not getattr(args, name):
            parser.error(f"--{name} (or SM_CHANNEL_{name.upper()}) is required")
    return args


def hyperparameters_from(args: argparse.Namespace) -> Hyperparameters:
    return Hyperparameters(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dims=tuple(int(width) for width in args.hidden_dims.split(",") if width),
        dropout=args.dropout,
        pos_weight_power=args.pos_weight_power,
        patience=args.patience,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    hp = hyperparameters_from(args)

    train_frame = read_split(Path(args.train))
    validation_frame = read_split(Path(args.validation))
    result = train(
        engineer(train_frame),
        train_frame[TARGET].to_numpy(),
        engineer(validation_frame),
        validation_frame[TARGET].to_numpy(),
        hp,
        log=lambda line: print(line, flush=True),
    )

    metadata = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "threshold": result.threshold,
        "threshold_rule": "max F1 on the validation split",
        "best_epoch": result.best_epoch,
        "validation": result.validation,
        "hyperparameters": {**hp.as_dict(), "hidden_dims": list(hp.hidden_dims)},
        "training_rows": int(len(train_frame)),
        "data_manifest": read_manifest(Path(args.train)),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    save_artifact(
        Path(args.model_dir), result.model, result.model_config, result.preprocessor, metadata
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_history.json").write_text(json.dumps(result.history, indent=2))
    print(f"saved model artifact to {args.model_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
