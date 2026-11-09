"""State of the research graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from alphax_agent.core.experiment import Experiment, ExperimentResult, Hypothesis
from alphax_agent.core.feedback import ExperimentFeedback, StepFailure
from alphax_agent.core.trace import Trace, TraceRecord


class ResearchState(TypedDict):
    """``history`` accumulates across loops; the other per-loop fields are cleared by the record step."""

    loop: int
    stop_at: int
    baseline: ExperimentResult | None
    history: Annotated[list[TraceRecord], operator.add]
    hypothesis: Hypothesis | None
    experiment: Experiment | None
    feedback: ExperimentFeedback | None
    failure: StepFailure | None


class StateView:
    """Typed accessors over a :class:`ResearchState`."""

    def __init__(self, state: ResearchState) -> None:
        self._state = state

    @property
    def loop(self) -> int:
        return self._state["loop"]

    @property
    def trace(self) -> Trace:
        return Trace(self._state.get("history", []), self._state.get("baseline"))

    def require_hypothesis(self) -> Hypothesis:
        hypothesis = self._state.get("hypothesis")
        if hypothesis is None:
            raise RuntimeError("No hypothesis in the state")
        return hypothesis

    def require_experiment(self) -> Experiment:
        experiment = self._state.get("experiment")
        if experiment is None:
            raise RuntimeError("No experiment in the state")
        return experiment

    @classmethod
    def initial(cls, history: list[TraceRecord], baseline: ExperimentResult | None, loops: int) -> ResearchState:
        start = max((record.loop for record in history), default=-1) + 1
        return ResearchState(
            loop=start,
            stop_at=start + loops,
            baseline=baseline,
            history=list(history),
            hypothesis=None,
            experiment=None,
            feedback=None,
            failure=None,
        )
