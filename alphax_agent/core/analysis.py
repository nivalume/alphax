"""What the analysis agent of every scenario returns, and the metric table it is shown."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, Field

from alphax_agent.core.experiment import ExperimentResult


class MetricComparison:
    """Side-by-side table of a result and the result it is compared with."""

    def __init__(
        self,
        candidate: ExperimentResult | None,
        reference: ExperimentResult | None,
        *,
        metrics: Sequence[str],
        labels: tuple[str, str] = ("current best", "this experiment"),
    ) -> None:
        self._candidate = candidate
        self._reference = reference
        self._labels = labels
        self._metrics = metrics

    def render(self) -> str:
        reference_label, candidate_label = self._labels
        rows = [f"| metric | {reference_label} | {candidate_label} | change |", "|---|---|---|---|"]
        for metric in self._metrics:
            old = self._reference.get(metric) if self._reference else None
            new = self._candidate.get(metric) if self._candidate else None
            change = f"{new - old:+.4f}" if old is not None and new is not None else "n/a"
            rows.append(f"| {metric} | {self._format(old)} | {self._format(new)} | {change} |")
        return "\n".join(rows)

    @staticmethod
    def _format(value: float | None) -> str:
        return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"


class FeedbackAnalysis(BaseModel):
    observations: str = Field(description="What the numbers show, compared with the current best")
    hypothesis_evaluation: str = Field(description="Whether and why the results support the hypothesis")
    hypothesis_supported: bool
    next_hypothesis: str = Field(description="The most promising hypothesis to test next, given this result")
    reason: str = Field(description="One-paragraph conclusion")
