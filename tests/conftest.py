from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_scoring.data import split_by_time, write_splits
from fraud_scoring.entrypoints.train import main as train_main
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


@pytest.fixture
def trained_model(processed_dir, tmp_path):
    model_dir = tmp_path / "jobs" / "job-1" / "model"
    train_main(
        [
            "--train", str(processed_dir / "train"),
            "--validation", str(processed_dir / "validation"),
            "--model-dir", str(model_dir),
            "--output-dir", str(tmp_path / "out"),
            "--epochs", "6",
            "--hidden-dims", "16,8",
        ]
    )  # fmt: skip
    return model_dir


class FakeEndpoint(BaseHTTPRequestHandler):
    """A stand-in for the real endpoint: flags amounts above 100, rejects negative ones (400)."""

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        rows = payload["instances"]
        if any(row["Amount"] < 0 for row in rows):
            self.send_response(400)
            self.end_headers()
            self.wfile.write(
                b'{"error": "Amount must be non-negative"}\nTraceback (most recent call last)'
            )
            return
        predictions = [
            {
                "fraud_probability": 0.99 if row["Amount"] > 100 else 0.01,
                "flagged": row["Amount"] > 100,
            }
            for row in rows
        ]
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({"predictions": predictions, "threshold": 0.5}).encode())


@pytest.fixture
def endpoint_url():
    httpd = HTTPServer(("127.0.0.1", 0), FakeEndpoint)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()
