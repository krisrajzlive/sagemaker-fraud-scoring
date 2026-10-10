from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_scoring.data import split_by_time, write_splits
from fraud_scoring.schema import RAW_COLUMNS, TARGET, V_COLUMNS


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    """Tests must never read the developer's .env or FRAUD_SCORING_* variables."""
    for name in list(__import__("os").environ):
        if name.startswith("FRAUD_SCORING_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def make_transactions(n: int, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """Synthetic transactions in the raw dataset's schema, with a learnable fraud signal."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < fraud_rate).astype("int8")
    values = rng.normal(size=(n, len(V_COLUMNS)))
    values[labels == 1, :4] += 3.0
    frame = pd.DataFrame(values, columns=V_COLUMNS)
    frame.insert(0, "Time", np.sort(rng.uniform(0, 172_800, size=n)).round())
    frame["Amount"] = rng.lognormal(mean=3.0, sigma=1.0, size=n).round(2)
    frame[TARGET] = labels
    return frame[RAW_COLUMNS]


@pytest.fixture
def transactions() -> pd.DataFrame:
    return make_transactions(6000)


@pytest.fixture
def processed_dir(tmp_path: Path, transactions: pd.DataFrame) -> Path:
    """Time-ordered train/validation/test parquet splits of the synthetic data."""
    out = tmp_path / "processed"
    write_splits(split_by_time(transactions), out, dataset_sha256="synthetic")
    return out
