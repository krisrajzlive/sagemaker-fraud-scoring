from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_scoring.features import SECONDS_PER_DAY, Preprocessor, engineer
from fraud_scoring.schema import FEATURE_NAMES, INPUT_FIELDS, V_COLUMNS
from tests.conftest import make_transactions


def _row(**overrides) -> pd.DataFrame:
    row = dict.fromkeys(INPUT_FIELDS, 0.0)
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineer_output_layout_and_values():
    features = engineer(_row(V1=1.5, V28=-2.0, Amount=9.0, Time=SECONDS_PER_DAY / 4))
    assert features.shape == (1, len(FEATURE_NAMES)) and features.dtype == np.float32
    assert features[0, 0] == pytest.approx(1.5)
    assert features[0, len(V_COLUMNS) - 1] == pytest.approx(-2.0)
    log_amount, tod_sin, tod_cos = features[0, -3:]
    assert log_amount == pytest.approx(np.log1p(9.0))
    assert tod_sin == pytest.approx(1.0) and tod_cos == pytest.approx(0.0, abs=1e-6)


def test_time_of_day_wraps_after_24_hours():
    a, b = engineer(_row(Time=1234.0)), engineer(_row(Time=1234.0 + SECONDS_PER_DAY))
    assert a[0, -2:] == pytest.approx(b[0, -2:], abs=1e-5)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (_row().drop(columns=["V7"]), "missing fields: V7"),
        (_row(Amount=-1.0), "non-negative"),
        (_row(V3=float("nan")), "non-finite"),
        (_row(V3=float("inf")), "non-finite"),
        (_row(Amount="abc"), "non-numeric"),
    ],
)
def test_engineer_rejects_bad_transactions(frame, message):
    with pytest.raises(ValueError, match=message):
        engineer(frame)


def test_preprocessor_standardises_with_training_statistics():
    train = engineer(make_transactions(2000, seed=1))
    pre = Preprocessor().fit(train)
    scaled = pre.transform(train)
    assert scaled.mean(axis=0) == pytest.approx(0.0, abs=1e-4)
    assert scaled.std(axis=0) == pytest.approx(1.0, abs=1e-3)
    other = engineer(make_transactions(500, seed=2))
    assert pre.transform(other).shape == other.shape


def test_preprocessor_leaves_constant_features_finite():
    constant = np.ones((10, 3), dtype=np.float32)
    scaled = Preprocessor().fit(constant).transform(constant)
    assert np.isfinite(scaled).all() and (scaled == 0).all()


def test_preprocessor_roundtrips_and_validates_shape_and_version():
    pre = Preprocessor().fit(engineer(make_transactions(300)))
    restored = Preprocessor.from_dict(pre.to_dict())
    sample = engineer(make_transactions(20, seed=5))
    assert np.array_equal(pre.transform(sample), restored.transform(sample))

    with pytest.raises(ValueError, match="expected a"):
        pre.transform(sample[:, :-1])
    with pytest.raises(ValueError, match="version"):
        Preprocessor.from_dict({**pre.to_dict(), "version": 99})
    with pytest.raises(ValueError, match="feature names"):
        Preprocessor.from_dict({**pre.to_dict(), "feature_names": ["x"]})
