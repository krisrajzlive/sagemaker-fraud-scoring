from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from fraud_scoring.cli import app
from fraud_scoring.data import split_by_time, write_splits

runner = CliRunner()


@pytest.fixture
def prepared(transactions):
    write_splits(split_by_time(transactions), Path("data/processed"), "synthetic")


def _session(endpoint_url: str, *lines: str):
    """Run `fraud-scoring try` typing the given lines; returns (exit code, folded output)."""
    result = runner.invoke(app, ["try", "--url", endpoint_url], input="\n".join(lines) + "\n")
    return result.exit_code, " ".join(result.output.split())


def test_it_scores_the_starting_transaction_and_explains_itself(prepared, endpoint_url):
    code, text = _session(endpoint_url, "quit")
    assert code == 0
    assert "typical legitimate transaction" in text
    assert "fraud probability 0.0100" in text
    assert "FLAGGED" not in text, "the typical transaction is not flagged"
    assert "random fraud" in text, "the help text lists what can be typed"


def test_changing_a_field_rescores_the_transaction(prepared, endpoint_url):
    code, text = _session(endpoint_url, "Amount=500", "quit")
    assert code == 0 and "FLAGGED as fraud" in text and "0.9900" in text


def test_set_keyword_hour_shortcut_and_show(prepared, endpoint_url):
    code, text = _session(endpoint_url, "set amount=500 hour=3", "show", "quit")
    assert code == 0
    assert "Amount" in text and "500.0000" in text
    assert "Time" in text and "10,800.0000" in text


def test_reset_returns_to_the_starting_point(prepared, endpoint_url):
    _, text = _session(endpoint_url, "Amount=500", "reset", "show", "quit")
    assert text.count("FLAGGED as fraud") == 1
    assert "no changes yet" in text


def test_pasting_json_sets_several_fields(prepared, endpoint_url):
    _, text = _session(endpoint_url, '{"Amount": 900, "Class": 1}', "quit")
    assert "FLAGGED as fraud" in text


def test_random_real_transactions_report_their_true_label(prepared, endpoint_url):
    _, text = _session(endpoint_url, "random fraud", "random legit", "quit")
    assert "loaded a real test-split transaction (actually FRAUD)" in text
    assert "loaded a real test-split transaction (actually legitimate)" in text


def test_typos_are_reported_and_the_session_carries_on(prepared, endpoint_url):
    code, text = _session(
        endpoint_url, "V99=1", "V14=abc", "Amount", "random sideways", "dance", "Amount=500", "quit"
    )
    assert code == 0
    assert "unknown field 'V99'" in text
    assert "V14 needs a number" in text
    assert "to change Amount, type Amount=VALUE" in text
    assert "use 'random fraud' or 'random legit'" in text
    assert "unknown command 'dance'" in text
    assert "FLAGGED as fraud" in text, "a good command after the mistakes still works"


def test_a_rejected_transaction_shows_the_endpoints_message_and_continues(prepared, endpoint_url):
    code, text = _session(endpoint_url, "Amount=-5", "Amount=50", "quit")
    assert code == 0
    assert "the endpoint rejected this transaction" in text
    assert "Amount must be non-negative" in text
    assert "Traceback" not in text, "only the message line is shown"
    assert text.rindex("fraud probability") > text.index("rejected"), "scoring resumes after a fix"


def test_end_of_input_ends_the_session_cleanly(prepared, endpoint_url):
    result = runner.invoke(app, ["try", "--url", endpoint_url], input="Amount=500\n")
    assert result.exit_code == 0


def test_square_brackets_in_input_do_not_break_the_output(prepared, endpoint_url):
    code, text = _session(endpoint_url, "V1=[red]", "quit")
    assert code == 0 and "needs a number" in text


def test_it_needs_prepared_data(endpoint_url):
    result = runner.invoke(app, ["try", "--url", endpoint_url], input="quit\n")
    assert result.exit_code == 1
    assert "data prepare" in " ".join(result.output.split())


def test_help_lists_try():
    assert "try" in runner.invoke(app, ["--help"]).output
