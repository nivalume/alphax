"""Hypothesis cards: the mechanism behind a test, written before any data is seen."""

from __future__ import annotations

import re

REQUIRED_TEXT = ("title", "mechanism", "why_exists", "counterparty", "expected_horizon", "failure_conditions")
EXPERIMENT_TYPES = ("factor_ts",)

_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_DATE = re.compile(r"\d{4}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}月|\d{4}年")
_MONTH = re.compile(
    # "may" and "march" are left out of the bare-word list: they are ordinary English.
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d|\b(january|february|april|june|"
    r"july|august|september|october|november|december)\b",
    re.IGNORECASE,
)


class CardError(ValueError):
    pass


def _text(card: dict) -> str:
    parts = [str(card.get(key, "")) for key in (*REQUIRED_TEXT, "expected_behavior", "rationale")]
    return " ".join(parts)


def regulate(card: dict, forbidden_terms: list[str]) -> list[str]:
    """Problems with a card; empty means it may be registered.

    Cards may not mention dates, years, months or named market events: the LLM's
    training data covers the evaluation periods, and naming what happened when is
    the most direct way that memory leaks into a hypothesis.
    """
    problems = [f"missing {key}" for key in REQUIRED_TEXT if not str(card.get(key, "")).strip()]
    if card.get("experiment_type") not in EXPERIMENT_TYPES:
        problems.append(f"experiment_type must be one of {EXPERIMENT_TYPES}")
    if card.get("direction") not in (1, -1):
        problems.append("direction must be 1 (long the factor) or -1 (short the factor), fixed before any data")
    ranges = card.get("param_ranges")
    if not isinstance(ranges, dict) or not ranges:
        problems.append("param_ranges must give a theory-based [low, high] for every parameter")
    else:
        for name, bounds in ranges.items():
            if not (isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(b, (int, float)) for b in bounds)):
                problems.append(f"param_ranges.{name} must be [low, high]")
            elif bounds[0] > bounds[1]:
                problems.append(f"param_ranges.{name} has low > high")
    text = _text(card)
    for pattern, what in ((_YEAR, "a year"), (_DATE, "a date"), (_MONTH, "a month")):
        match = pattern.search(text)
        if match:
            problems.append(f"mentions {what} ({match.group(0)!r}); describe the mechanism without dates")
    lowered = text.lower()
    for term in forbidden_terms:
        if re.search(rf"(?<![a-z]){re.escape(term.lower())}(?![a-z])", lowered):
            problems.append(f"mentions the named event {term!r}; describe the mechanism, not the episode")
    return problems
