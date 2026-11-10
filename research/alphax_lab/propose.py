"""Hypothesis cards and specs from an LLM that never sees a result.

Inputs are only: the direction (prompt text, PDF text, or nothing), the symbols, and the
titles and templates already registered (for de-duplication). No metric, verdict, or
report ever enters a prompt; `research/tests/test_pipeline.py::test_proposals_never_see_results`
asserts this on the transcript.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from alphax_lab.llm import LLM
from alphax_lab.protocol import cards as card_rules
from alphax_lab.protocol import experiment_type

FAMILIES = ("trend", "mean_reversion", "breakout", "volatility", "volume_price", "range_position", "seasonality")

CARD_SCHEMA = """Each card is a JSON object:
{"title": short name,
 "experiment_type": "factor_ts",
 "mechanism": what market behaviour produces the effect,
 "why_exists": why it is not arbitraged away (risk premium, constraint, behavioural bias...),
 "counterparty": who is on the losing side and why they keep trading,
 "expected_horizon": holding period implied by the mechanism, in hours/days/weeks,
 "expected_behavior": {"majors": expectation on BTC/ETH-like assets, "alts": expectation on small caps},
 "failure_conditions": when the mechanism should stop working,
 "direction": 1 if a HIGH factor value should precede POSITIVE returns, -1 if the opposite,
 "param_ranges": {"<param>": [low, high], ...} for every expression and signal parameter, justified by
                 the mechanism (e.g. a multi-week trend implies lookbacks of 168-2160 one-hour bars),
 "rationale": how the factor expression captures the mechanism}
Rules: describe mechanisms, never episodes. Do not write years, dates, month names, 4-digit numbers,
or named market events (exchange collapses, halvings, ETF decisions, elections...). The direction and the
ranges are fixed before any data is seen and cannot be changed afterwards."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _direction_block(prompt: str | None, pdf_notes: list[dict] | None) -> tuple[str, dict]:
    if pdf_notes:
        text = "\n\n".join(f"(pages {n['pages'][0]}-{n['pages'][1]})\n{n['text']}" for n in pdf_notes)
        source = {"kind": "pdf", "pages": [n["pages"] for n in pdf_notes]}
        return f"Research direction: the mechanisms described in this literature.\n\n{text}", source
    if prompt:
        return f"Research direction from the researcher:\n{prompt}", {"kind": "prompt", "text": prompt}
    return (
        "No direction was given. Cover distinct mechanisms across these families: " + ", ".join(FAMILIES),
        {"kind": "autonomous"},
    )


def pdf_mechanisms(llm: LLM, chunks: list[dict], path: str) -> list[dict]:
    """Condense each PDF chunk to the mechanisms it argues for, with page references."""
    notes = []
    for chunk in chunks:
        reply = llm.chat_json(
            "Extract the trading mechanisms this text argues for. For each: the mechanism, why it should exist, "
            "the horizon, and the pages. Leave out dates, sample periods and reported performance numbers. "
            'Return JSON {"mechanisms": [{"mechanism": str, "why": str, "horizon": str, "pages": str}]}.',
            chunk["text"],
        )
        body = json.dumps(reply.get("mechanisms", []), ensure_ascii=False)
        notes.append({"pages": chunk["pages"], "text": body, "path": path})
    return notes


def generate(
    llm: LLM,
    *,
    symbols: list[str],
    n: int,
    tried: dict[str, str],
    policy: dict,
    prompt: str | None = None,
    pdf_notes: list[dict] | None = None,
    pdf_path: str | None = None,
    attempts: int = 3,
) -> list[dict]:
    """Returns pending items {id, card, spec, problems}; problems is empty when both are valid."""
    direction, source = _direction_block(prompt, pdf_notes)
    if pdf_path:
        source["path"] = pdf_path
    kind = experiment_type.get("factor_ts")
    forbidden = policy["card"]["forbidden_terms"]
    tried_block = json.dumps(tried, ensure_ascii=False) if tried else "none yet"
    reply = llm.chat_json(
        f"You are a quantitative researcher writing pre-registered hypothesis cards for {', '.join(symbols)} "
        f"perpetual futures on 1h bars.\n{CARD_SCHEMA}\n\n{kind.spec_prompt()}\n\n"
        'Return JSON {"cards": [{"card": card, "spec": spec}, ...]}.',
        f"{direction}\n\nWrite {n} cards with distinct mechanisms. Already registered (title -> template), do not "
        f"repeat them:\n{tried_block}",
    )
    items = []
    for raw in (reply.get("cards") or [])[:n]:
        card = dict(raw.get("card") or {}, symbols=symbols, source=source, experiment_type="factor_ts")
        spec = raw.get("spec") or {}
        for attempt in range(attempts + 1):
            problems = card_rules.regulate(card, forbidden)
            normalized, spec_problems = (None, []) if problems else kind.regulate(spec, card, tried, policy)
            problems += spec_problems
            if not problems or attempt == attempts:
                break
            # Only format errors go back to the model — never data or results.
            fixed = llm.chat_json(
                f"{CARD_SCHEMA}\n\n{kind.spec_prompt()}\n\nFix the card and spec so these problems disappear. Keep the "
                'mechanism. Return JSON {"card": card, "spec": spec}.',
                json.dumps({"card": card, "spec": spec, "problems": problems}, ensure_ascii=False),
            )
            card = dict(fixed.get("card") or card, symbols=symbols, source=source, experiment_type="factor_ts")
            spec = fixed.get("spec") or spec
        item_id = hashlib.sha256(json.dumps([card.get("title"), _now()]).encode()).hexdigest()[:8]
        items.append({"id": item_id, "created": _now(), "card": card, "spec": normalized or spec, "problems": problems})
        if normalized is not None:
            title = card.get("title")
            if not isinstance(title, str):
                raise ValueError("normalized card is missing a title")
            tried = {**tried, title: normalized["expression"]}
    return items


def repair(llm: LLM, item: dict, error: str) -> dict:
    """After a contract-check failure: the model sees the spec and the error, nothing else."""
    kind = experiment_type.get("factor_ts")
    fixed = llm.chat_json(
        f"{kind.spec_prompt()}\n\nThe spec failed a synthetic-data contract check. Fix the expression or parameter "
        "values only as far as the error requires; keep the card's mechanism and ranges. "
        'Return JSON {"spec": spec}.',
        json.dumps({"card_title": item["card"]["title"], "param_ranges": item["card"]["param_ranges"],
                    "spec": item["spec"], "error": error}, ensure_ascii=False),
    )
    return dict(item, spec=fixed.get("spec") or item["spec"])
