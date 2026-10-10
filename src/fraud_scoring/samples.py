"""Real transactions drawn from the held-out test split, for trying out an endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fraud_scoring.data import read_split
from fraud_scoring.schema import INPUT_FIELDS, TARGET


def sample_transactions(
    test_dir: Path, *, count: int, frauds: int, seed: int = 7
) -> tuple[list[dict[str, Any]], list[int]]:
    """Pick ``count`` test transactions, ``frauds`` of them fraudulent (fewer if none remain).

    Returns the request instances (scoring fields only, never the label) and the true labels in
    the same order, so a caller can compare predictions with the truth without leaking it.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    frame = read_split(test_dir)
    rng = np.random.default_rng(seed)
    fraud_rows = np.flatnonzero(frame[TARGET].to_numpy() == 1)
    legit_rows = np.flatnonzero(frame[TARGET].to_numpy() == 0)
    n_fraud = min(max(frauds, 0), count, len(fraud_rows))
    picks = np.concatenate(
        [
            rng.choice(fraud_rows, size=n_fraud, replace=False),
            rng.choice(legit_rows, size=count - n_fraud, replace=False),
        ]
    )
    rng.shuffle(picks)
    chosen = frame.iloc[picks]
    return chosen[INPUT_FIELDS].to_dict(orient="records"), chosen[TARGET].astype(int).tolist()
