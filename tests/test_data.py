from __future__ import annotations

import json

import pandas as pd
import pytest

from fraud_scoring.data import (
    DataIntegrityError,
    clean,
    describe_splits,
    download_dataset,
    read_manifest,
    read_split,
    sha256_of,
    split_by_time,
    verify_dataset,
    write_splits,
)
from fraud_scoring.schema import SPLITS, TARGET
from tests.conftest import make_transactions


def test_clean_drops_exact_duplicates_and_orders_by_time():
    frame = make_transactions(200)
    shuffled = pd.concat([frame, frame.iloc[:15]]).sample(frac=1, random_state=3)
    cleaned = clean(shuffled)
    assert len(cleaned) == len(frame)
    assert cleaned["Time"].is_monotonic_increasing
    assert cleaned[TARGET].dtype == "int8"


def test_clean_rejects_missing_values_and_bad_labels():
    frame = make_transactions(50)
    broken = frame.copy()
    broken.loc[3, "V1"] = None
    with pytest.raises(DataIntegrityError, match="missing"):
        clean(broken)
    relabelled = frame.copy()
    relabelled.loc[0, TARGET] = 2
    with pytest.raises(DataIntegrityError, match="0 or 1"):
        clean(relabelled)


def test_split_is_chronological_and_complete(transactions):
    splits = split_by_time(clean(transactions))
    assert sum(len(part) for part in splits.values()) == len(clean(transactions))
    assert splits["train"]["Time"].max() < splits["validation"]["Time"].min()
    assert splits["validation"]["Time"].max() < splits["test"]["Time"].min()
    assert len(splits["train"]) / len(clean(transactions)) == pytest.approx(0.70, abs=0.01)


def test_split_never_separates_rows_sharing_a_timestamp():
    frame = make_transactions(100)
    frame["Time"] = (frame.index // 10).astype(float)  # blocks of ten identical timestamps
    splits = split_by_time(frame, train_fraction=0.55, validation_fraction=0.2)
    train_times, validation_times = set(splits["train"]["Time"]), set(splits["validation"]["Time"])
    assert not train_times & validation_times
    assert not validation_times & set(splits["test"]["Time"])


@pytest.mark.parametrize("fractions", [(0, 0.1), (0.9, 0.1), (0.6, 1.2)])
def test_split_validates_fractions(fractions):
    with pytest.raises(ValueError):
        split_by_time(make_transactions(30), *fractions)


def test_write_and_read_splits_roundtrip(tmp_path, transactions):
    splits = split_by_time(transactions)
    manifest = write_splits(splits, tmp_path / "out", dataset_sha256="abc")
    for name in SPLITS:
        restored = read_split(tmp_path / "out" / name)
        pd.testing.assert_frame_equal(restored, splits[name])
        assert read_manifest(tmp_path / "out" / name)["dataset_sha256"] == "abc"
    assert manifest == describe_splits(splits, "abc")
    assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == manifest


def test_rewriting_splits_removes_stale_parquet_files(tmp_path, transactions):
    out = tmp_path / "out"
    write_splits(split_by_time(transactions), out, "v1")
    (out / "train" / "part-9.parquet").write_bytes(b"stale")
    write_splits(split_by_time(transactions), out, "v2")
    assert len(read_split(out / "train")) == len(split_by_time(transactions)["train"])


def test_read_split_requires_parquet_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_split(tmp_path)


def test_download_verifies_checksum_and_reuses_verified_copy(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n")
    url = source.as_uri()
    digest = sha256_of(source)

    path = download_dataset(tmp_path / "raw", url=url, expected_sha256=digest)
    assert path.read_text() == source.read_text()

    source.unlink()  # a verified local copy must not need the network again
    assert download_dataset(tmp_path / "raw", url=url, expected_sha256=digest) == path


def test_download_rejects_tampered_content_and_leaves_no_partial_file(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("a,b\n1,2\n")
    with pytest.raises(DataIntegrityError, match="expected"):
        download_dataset(tmp_path / "raw", url=source.as_uri(), expected_sha256="0" * 64)
    assert list((tmp_path / "raw").iterdir()) == []


def test_verify_dataset_detects_modified_file(tmp_path):
    path = tmp_path / "creditcard.csv"
    path.write_text("x")
    with pytest.raises(DataIntegrityError):
        verify_dataset(path)
