"""Research history: what was tried, what it produced and what was kept."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, SerializeAsAny

from alphax.core.experiment import Experiment, ExperimentResult, Hypothesis
from alphax.core.feedback import ExperimentFeedback


class TraceRecord(BaseModel):
    """One completed research loop."""

    loop: int
    hypothesis: Hypothesis | None = None
    experiment: SerializeAsAny[Experiment] | None = None
    feedback: ExperimentFeedback

    def render(self) -> str:
        parts = [f"## Loop {self.loop}"]
        if self.hypothesis is not None:
            parts.append(self.hypothesis.render())
        if self.experiment is not None:
            parts.append("Tasks:\n" + self.experiment.render_tasks())
            if self.experiment.result is not None:
                parts.append("Result:\n" + self.experiment.result.render())
        parts.append("Feedback:\n" + self.feedback.render())
        return "\n".join(parts)


class Trace:
    """Read-only view over the records of a run.

    The records live in the workflow state; a trace is rebuilt from them whenever a step needs
    to reason about the history, so there is a single source of truth.
    """

    def __init__(self, records: Sequence[TraceRecord] = (), baseline: ExperimentResult | None = None) -> None:
        self._records = list(records)
        self._baseline = baseline

    @property
    def records(self) -> list[TraceRecord]:
        return list(self._records)

    @property
    def baseline(self) -> ExperimentResult | None:
        return self._baseline

    def __len__(self) -> int:
        return len(self._records)

    def accepted(self) -> list[TraceRecord]:
        """Records promoted to best, oldest first; together they form the current best solution."""
        return [record for record in self._records if record.feedback.decision and record.experiment is not None]

    def sota(self) -> TraceRecord | None:
        accepted = self.accepted()
        return accepted[-1] if accepted else None

    def reference_result(self) -> ExperimentResult | None:
        """The result a new experiment has to beat: the latest best, else the baseline."""
        sota = self.sota()
        if sota is not None and sota.experiment is not None and sota.experiment.result is not None:
            return sota.experiment.result
        return self._baseline

    def recent(self, limit: int) -> list[TraceRecord]:
        return self._records[-limit:] if limit > 0 else []

    def consecutive_failures(self) -> int:
        count = 0
        for record in reversed(self._records):
            if not record.feedback.failed:
                break
            count += 1
        return count

    def render_history(self, limit: int = 8) -> str:
        records = self.recent(limit)
        if not records:
            return "No experiment has been run yet."
        return "\n\n".join(record.render() for record in records)
