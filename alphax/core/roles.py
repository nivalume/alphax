"""The five roles of a research loop.

A scenario plugs into the workflow by providing one object per role. LLM-backed roles are
implemented as agents; deterministic roles (such as running a backtest) are plain classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from alphax.core.experiment import Experiment, ExperimentResult, Hypothesis
from alphax.core.feedback import ExperimentFeedback
from alphax.core.trace import Trace


class HypothesisProposer(ABC):
    """Proposes the next hypothesis from the research history."""

    @abstractmethod
    def propose(self, trace: Trace) -> Hypothesis: ...


class ExperimentDesigner(ABC):
    """Turns a hypothesis into concrete, implementable tasks."""

    @abstractmethod
    def design(self, hypothesis: Hypothesis, trace: Trace) -> Experiment: ...


class Developer(ABC):
    """Implements the tasks of an experiment and attaches the selected implementations."""

    @abstractmethod
    def develop(self, experiment: Experiment, trace: Trace) -> Experiment: ...


class Runner(ABC):
    """Runs a developed experiment and attaches its result."""

    @abstractmethod
    def run(self, experiment: Experiment, trace: Trace) -> Experiment: ...

    def establish_baseline(self) -> ExperimentResult | None:
        """Result of the reference setup before any research; ``None`` if there is none."""
        return None


class Summarizer(ABC):
    """Judges a run experiment against the history."""

    @abstractmethod
    def summarize(self, experiment: Experiment, trace: Trace) -> ExperimentFeedback: ...
