"""Logic behind ``fraud-scoring try``: edit one transaction by hand and see how the score moves.

The model only understands the dataset's numeric fields (``Time``, ``V1``..``V28``, ``Amount``;
the V fields are anonymised PCA components), so "any input" means any numbers for those fields.
"""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field

import pandas as pd

from fraud_scoring.schema import INPUT_FIELDS, TARGET

FIELD_BY_LOWER_NAME = {name.lower(): name for name in INPUT_FIELDS}
SECONDS_PER_HOUR = 3600.0

HELP = """Change any field and the transaction is scored again. Examples:
  V14=-8 V12=-7       set fields (V1..V28, Amount, Time; any letter case)
  Amount=2500 hour=3  'hour' is a shortcut for the time of day
  {"V14": -9}         paste a JSON object to set several fields
  random fraud        load a real fraud from the held-out test split
  random legit        load a real legitimate transaction
  show                list the fields you changed ('show all': every field)
  reset               go back to where you started
  help, quit"""


class CommandError(ValueError):
    """A command the user typed could not be understood."""


def _field_name(raw: str) -> str:
    key = raw.strip().lower()
    if key in FIELD_BY_LOWER_NAME:
        return FIELD_BY_LOWER_NAME[key]
    suggestion = difflib.get_close_matches(key, list(FIELD_BY_LOWER_NAME) + ["hour"], n=1)
    hint = (
        f" (did you mean {FIELD_BY_LOWER_NAME.get(suggestion[0], suggestion[0])}?)"
        if suggestion
        else ""
    )
    raise CommandError(f"unknown field {raw!r}{hint}; fields are Time, V1..V28, Amount, hour")


def _number(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError:
        raise CommandError(f"{name} needs a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise CommandError(f"{name} must be a finite number")
    return value


def parse_assignments(text: str) -> dict[str, float]:
    """``'V14=-8 Amount=2500 hour=3'`` -> ``{'V14': -8.0, 'Amount': 2500.0, 'Time': 10800.0}``."""
    assignments: dict[str, float] = {}
    for token in text.replace(",", " ").split():
        name, separator, raw_value = token.partition("=")
        if not separator or not raw_value:
            raise CommandError(f"expected NAME=VALUE, got {token!r}")
        if name.strip().lower() == "hour":
            hour = _number(raw_value, "hour")
            if not 0 <= hour < 24:
                raise CommandError("hour must be from 0 up to (not including) 24")
            assignments["Time"] = hour * SECONDS_PER_HOUR
        else:
            field_name = _field_name(name)
            assignments[field_name] = _number(raw_value, field_name)
    if not assignments:
        raise CommandError("nothing to set")
    return assignments


def parse_json_fields(text: str) -> dict[str, float]:
    """A pasted JSON object; keys other than the scoring fields (e.g. a label) are ignored."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise CommandError(f"not valid JSON: {exc}") from None
    if not isinstance(payload, dict):
        raise CommandError('expected a JSON object such as {"V14": -8}')
    assignments = {
        FIELD_BY_LOWER_NAME[key.lower()]: _number(str(value), key)
        for key, value in payload.items()
        if key.lower() in FIELD_BY_LOWER_NAME
    }
    if not assignments:
        raise CommandError("the JSON object has none of the scoring fields")
    return assignments


def typical_transaction(frame: pd.DataFrame) -> dict[str, float]:
    """A synthetic 'average' legitimate transaction: field-wise median, at midday.

    It is a convenient starting point, not a real transaction.
    """
    legit = frame[frame[TARGET] == 0]
    typical = {name: float(value) for name, value in legit[INPUT_FIELDS].median().items()}
    typical["Time"] = 12 * SECONDS_PER_HOUR
    return typical


@dataclass
class Session:
    """The transaction being edited, and the one it started from."""

    reference: dict[str, float]
    origin: str
    current: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.current = dict(self.reference)

    def apply(self, assignments: dict[str, float]) -> None:
        self.current.update(assignments)

    def load(self, transaction: dict[str, float], origin: str) -> None:
        self.reference = {name: float(transaction[name]) for name in INPUT_FIELDS}
        self.origin = origin
        self.current = dict(self.reference)

    def reset(self) -> None:
        self.current = dict(self.reference)

    def changed(self) -> dict[str, tuple[float, float]]:
        """Fields that differ from the starting point: ``name -> (before, now)``."""
        return {
            name: (self.reference[name], self.current[name])
            for name in INPUT_FIELDS
            if self.current[name] != self.reference[name]
        }


def format_score(probability: float, threshold: float, width: int = 30) -> str:
    """A text gauge: filled up to the probability, ``|`` at the decision threshold."""
    filled = round(min(max(probability, 0.0), 1.0) * width)
    marker = min(int(threshold * width), width - 1)
    cells = ["#" if i < filled else "." for i in range(width)]
    cells[marker] = "|"
    return "[" + "".join(cells) + "]"
