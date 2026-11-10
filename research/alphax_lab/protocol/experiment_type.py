"""The plug-in boundary between the research protocol and a kind of experiment.

The protocol (cards, registration, ledger counts, gate thresholds, OOS batches,
forward tracking) is the same for every type. A type decides what a spec looks
like, how one run executes, which perturbations it reports, and which of its
numbers the gate judges. Only `factor_ts` exists; add a type when a card needs one.
"""

from __future__ import annotations

from typing import Protocol


class ExperimentType(Protocol):
    name: str

    def spec_prompt(self) -> str:
        """What the LLM needs to translate a card into this type's spec."""
        ...

    def regulate(self, raw_spec: dict, card: dict, tried: dict[str, str], policy: dict) -> tuple[dict | None, list[str]]:
        """Normalized spec, or None with the reasons it is not valid."""
        ...

    def contract_check(self, spec: dict, card: dict, policy: dict) -> str | None:
        """Error message from a synthetic-data run, or None."""
        ...

    def run(self, spec: dict, card: dict, policy: dict, window: str, *, control: str | None = None) -> dict:
        """One evaluation on `window` ("in_sample", "out_of_sample" or "forward:<start>")."""
        ...

    def gate(self, result: dict, policy: dict, tests: int) -> list[dict]:
        """In-sample checks: [{name, passed, value, threshold}]."""
        ...

    def oos_gate(self, result: dict, policy: dict, tests: int) -> list[dict]:
        """Out-of-sample checks."""
        ...

    def report_sections(self, result: dict) -> list[tuple[str, str, bool]]:
        """(title, markdown, gated) sections for the fixed report template."""
        ...


def get(name: str) -> ExperimentType:
    if name == "factor_ts":
        from alphax_lab.types.factor_ts import FactorTS

        return FactorTS()
    raise KeyError(f"unknown experiment type {name!r}")
