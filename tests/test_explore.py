from __future__ import annotations

import pytest

from fraud_scoring.explore import (
    CommandError,
    Session,
    format_score,
    parse_assignments,
    parse_json_fields,
    typical_transaction,
)
from fraud_scoring.schema import INPUT_FIELDS, TARGET
from tests.conftest import make_transactions


def test_assignments_are_case_insensitive_and_accept_commas():
    assert parse_assignments("v14=-8 AMOUNT=2500, v1=0.5") == {
        "V14": -8.0,
        "Amount": 2500.0,
        "V1": 0.5,
    }


def test_hour_is_a_shortcut_for_the_time_field():
    assert parse_assignments("hour=3") == {"Time": 3 * 3600.0}
    assert parse_assignments("Hour=13.5") == {"Time": 13.5 * 3600.0}


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("", "nothing to set"),
        ("V14", "NAME=VALUE"),
        ("V14=", "NAME=VALUE"),
        ("V14=abc", "needs a number"),
        ("V14=nan", "finite"),
        ("Amount=inf", "finite"),
        ("hour=24", "hour must be"),
        ("hour=-1", "hour must be"),
        ("V99=1", "unknown field"),
        ("Class=1", "unknown field"),
    ],
)
def test_bad_assignments_explain_themselves(text, fragment):
    with pytest.raises(CommandError, match=fragment):
        parse_assignments(text)


def test_unknown_field_suggests_the_closest_name():
    with pytest.raises(CommandError, match="did you mean V14"):
        parse_assignments("V140=1")
    with pytest.raises(CommandError, match="did you mean Amount"):
        parse_assignments("amout=1")


def test_json_fields_ignore_labels_and_unrelated_keys():
    parsed = parse_json_fields('{"v14": -9, "Amount": 40, "Class": 1, "comment": "x"}')
    assert parsed == {"V14": -9.0, "Amount": 40.0}


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("{oops", "not valid JSON"),
        ("[1, 2]", "JSON object"),
        ('{"comment": "x"}', "none of the scoring fields"),
        ('{"V14": "abc"}', "needs a number"),
    ],
)
def test_bad_json_is_reported(text, fragment):
    with pytest.raises(CommandError, match=fragment):
        parse_json_fields(text)


def test_typical_transaction_is_the_legit_median_at_midday():
    frame = make_transactions(2000, fraud_rate=0.2, seed=3)
    typical = typical_transaction(frame)
    assert set(typical) == set(INPUT_FIELDS)
    assert typical["Time"] == 12 * 3600.0
    legit_median = frame[frame[TARGET] == 0]["V1"].median()
    assert typical["V1"] == pytest.approx(legit_median)
    # the planted fraud signal lives in V1..V4; frauds would pull the median visibly upwards
    assert typical["V1"] < frame[frame[TARGET] == 1]["V1"].median() - 1.0


def test_session_tracks_changes_and_resets():
    start = {name: 0.0 for name in INPUT_FIELDS}
    session = Session(start, origin="zeros")
    assert session.changed() == {}

    session.apply({"V14": -8.0, "Amount": 5.0})
    assert session.changed() == {"V14": (0.0, -8.0), "Amount": (0.0, 5.0)}
    session.apply({"V14": 0.0})
    assert list(session.changed()) == ["Amount"], "setting a field back is not a change"

    session.reset()
    assert session.changed() == {} and session.current == start
    assert start["Amount"] == 0.0, "the reference must never be modified"


def test_loading_a_transaction_replaces_the_starting_point():
    session = Session({name: 0.0 for name in INPUT_FIELDS}, origin="zeros")
    session.apply({"V1": 3.0})
    loaded = {name: 1.0 for name in INPUT_FIELDS}
    session.load(loaded, origin="a real one")
    assert session.origin == "a real one" and session.changed() == {}
    session.apply({"V1": 2.0})
    session.reset()
    assert session.current["V1"] == 1.0


def test_gauge_fills_to_the_probability_and_marks_the_threshold():
    gauge = format_score(0.5, 0.9, width=20)
    assert gauge == "[" + "#" * 10 + "." * 8 + "|" + "." + "]"
    assert len(format_score(0.123, 0.5)) == 32
    assert format_score(0.0, 0.5, width=10).count("#") == 0
    assert format_score(1.0, 0.5, width=10).count("#") == 9  # the threshold mark takes one cell
    assert format_score(7.0, 0.5, width=10).count("#") == 9, "clamped above 1"
    assert format_score(-3.0, 0.5, width=10).count("#") == 0, "clamped below 0"
