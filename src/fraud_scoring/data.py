"""Dataset download, cleaning, time-ordered splitting and parquet I/O.

Data: the ULB "Credit Card Fraud Detection" dataset (European card transactions over two days,
473 frauds among 283,726 unique rows). V1..V28 are PCA components of undisclosed original
features. Dal Pozzolo et al., "Calibrating Probability with Undersampling for Unbalanced
Classification", IEEE SSCI 2015. Released by the ULB Machine Learning Group under the Database
Contents License (DbCL) v1.0. The file is downloaded on demand and never committed.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_scoring.schema import RAW_COLUMNS, SPLITS, TARGET

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATASET_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
DATASET_BYTES = 150_828_752
RAW_FILENAME = "creditcard.csv"
MANIFEST_FILENAME = "manifest.json"
PART_FILENAME = "part-0.parquet"


class DataIntegrityError(RuntimeError):
    """The dataset on disk is not the file this code was written and tested against."""


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def verify_dataset(path: Path, expected_sha256: str = DATASET_SHA256) -> None:
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise DataIntegrityError(
            f"{path} has sha256 {actual}, expected {expected_sha256}. "
            "Delete it and download again, or update DATASET_SHA256 if the source changed."
        )


def download_dataset(
    dest_dir: Path,
    *,
    url: str = DATASET_URL,
    expected_sha256: str = DATASET_SHA256,
    force: bool = False,
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download the CSV into ``dest_dir`` and verify its checksum. Reuses a verified copy."""
    dest = dest_dir / RAW_FILENAME
    if dest.exists() and not force:
        verify_dataset(dest, expected_sha256)
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "fraud-scoring/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:  # noqa: S310
        total = int(response.headers.get("Content-Length") or 0) or None
        received = 0
        while block := response.read(1 << 20):
            out.write(block)
            received += len(block)
            if on_progress:
                on_progress(received, total)
    try:
        verify_dataset(partial, expected_sha256)
    except DataIntegrityError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return dest


def load_raw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise DataIntegrityError(f"{path} is missing columns: {', '.join(missing)}")
    return frame[RAW_COLUMNS]


def clean(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, drop exact duplicate rows and order by time.

    The raw file contains 1,081 exact duplicate transactions. Left in, a duplicate can land in
    both the training and the held-out split and inflate the test score.
    """
    if frame[RAW_COLUMNS].isna().any().any():
        raise DataIntegrityError("dataset contains missing values")
    if not frame[TARGET].isin([0, 1]).all():
        raise DataIntegrityError(f"{TARGET} must be 0 or 1")
    deduplicated = frame.drop_duplicates()
    ordered = deduplicated.sort_values("Time", kind="stable").reset_index(drop=True)
    ordered[TARGET] = ordered[TARGET].astype("int8")
    return ordered


def _advance_to_time_boundary(times: np.ndarray, index: int) -> int:
    """Move a cut point forward so that rows sharing one timestamp are never split apart."""
    while 0 < index < len(times) and times[index] == times[index - 1]:
        index += 1
    return index


def split_by_time(
    frame: pd.DataFrame, train_fraction: float = 0.70, validation_fraction: float = 0.15
) -> dict[str, pd.DataFrame]:
    """Chronological split: train on the past, validate and test on progressively later data.

    A random split would let the model see transactions from the same hours it is tested on,
    which overstates how well it would score tomorrow's traffic.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave room for a test split")
    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    times = ordered["Time"].to_numpy()
    first = _advance_to_time_boundary(times, int(len(ordered) * train_fraction))
    second = _advance_to_time_boundary(
        times, int(len(ordered) * (train_fraction + validation_fraction))
    )
    return {
        "train": ordered.iloc[:first].reset_index(drop=True),
        "validation": ordered.iloc[first:second].reset_index(drop=True),
        "test": ordered.iloc[second:].reset_index(drop=True),
    }


def describe_splits(splits: dict[str, pd.DataFrame], dataset_sha256: str) -> dict[str, Any]:
    manifest: dict[str, Any] = {"dataset_sha256": dataset_sha256, "splits": {}}
    for name in SPLITS:
        part = splits[name]
        manifest["splits"][name] = {
            "rows": int(len(part)),
            "frauds": int(part[TARGET].sum()),
            "fraud_rate": float(part[TARGET].mean()),
            "time_start": float(part["Time"].min()),
            "time_end": float(part["Time"].max()),
        }
    return manifest


def write_splits(
    splits: dict[str, pd.DataFrame], out_dir: Path, dataset_sha256: str
) -> dict[str, Any]:
    """Write each split as ``<out_dir>/<split>/part-0.parquet`` (one SageMaker channel each)."""
    manifest = describe_splits(splits, dataset_sha256)
    for name in SPLITS:
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale in split_dir.glob("*.parquet"):
            stale.unlink()
        splits[name].to_parquet(split_dir / PART_FILENAME, index=False)
        (split_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    (out_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    return manifest


def read_split(directory: Path) -> pd.DataFrame:
    """Read every parquet file in a channel directory."""
    parts = sorted(directory.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no parquet files in {directory}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)


def read_manifest(directory: Path) -> dict[str, Any] | None:
    path = directory / MANIFEST_FILENAME
    return json.loads(path.read_text()) if path.exists() else None
