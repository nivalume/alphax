"""Nodes and routes of the research graph.

Each node wraps one research role. A recoverable failure inside a node does not stop the
run: the node returns a :class:`StepFailure`, the graph jumps to the summarize step, and the
loop is recorded as failed so the proposer can learn from it.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from langgraph.graph import END

from alphax_agent.config.settings import WorkflowSettings
from alphax_agent.core.exceptions import AlphaXError
from alphax_agent.core.feedback import ExperimentFeedback, StepFailure
from alphax_agent.core.roles import Developer, ExperimentDesigner, HypothesisProposer, Runner, Summarizer
from alphax_agent.core.text import Excerpt
from alphax_agent.core.trace import Trace, TraceRecord
from alphax_agent.llm.context import call_context
from alphax_agent.workflow.artifacts import ArtifactStore
from alphax_agent.workflow.state import ResearchState, StateView

logger = logging.getLogger(__name__)


class ResearchNode(ABC):
    name: ClassVar[str]
    recoverable: ClassVar[bool] = True

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        view = StateView(state)
        started = time.monotonic()
        logger.info("[loop %d] %s started", view.loop, self.name)
        try:
            update = self.execute(view)
        except Exception as exc:
            if not self.recoverable:
                raise
            if isinstance(exc, AlphaXError):
                logger.warning("[loop %d] %s failed: %s: %s", view.loop, self.name, type(exc).__name__, exc)
            else:
                logger.exception("[loop %d] %s raised an unexpected error", view.loop, self.name)
            failure = StepFailure(
                stage=self.name, error_type=type(exc).__name__, message=Excerpt.middle(str(exc), 4000)
            )
            return {"failure": failure}
        logger.info("[loop %d] %s finished in %.1fs", view.loop, self.name, time.monotonic() - started)
        return update

    @abstractmethod
    def execute(self, view: StateView) -> dict[str, Any]: ...


class TrackedStep:
    """A node of the research graph, observed.

    Tags every model call made inside the node with its loop and stage (see :mod:`alphax_agent.llm.context`)
    and appends the step's start, finish or failure, with its duration, to the run's ``events.jsonl``. The node
    itself does not know it is being observed.
    """

    def __init__(self, node: ResearchNode, store: ArtifactStore) -> None:
        self._node = node
        self._store = store

    @property
    def name(self) -> str:
        return self._node.name

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        loop, stage = state["loop"], self._node.name
        self._store.append_event(loop=loop, stage=stage, event="started")
        started = time.monotonic()
        try:
            with call_context(loop=loop, stage=stage):
                update = self._node(state)
        except Exception as exc:  # a node that does not recover: the run stops, the event says where
            self._store.append_event(
                loop=loop,
                stage=stage,
                event="failed",
                seconds=round(time.monotonic() - started, 3),
                error_type=type(exc).__name__,
                message=Excerpt.middle(str(exc), 4000),
            )
            raise
        seconds = round(time.monotonic() - started, 3)
        failure = update.get("failure")
        if failure is None:
            self._store.append_event(loop=loop, stage=stage, event="finished", seconds=seconds)
        else:
            self._store.append_event(
                loop=loop,
                stage=stage,
                event="failed",
                seconds=seconds,
                error_type=failure.error_type,
                message=failure.message,
            )
        return update


class BaselineNode(ResearchNode):
    """Runs the reference setup once per run; a failure here means the environment is broken."""

    name = "baseline"
    recoverable = False

    def __init__(self, runner: Runner, store: ArtifactStore) -> None:
        self._runner = runner
        self._store = store

    def execute(self, view: StateView) -> dict[str, Any]:
        if view.trace.baseline is not None:
            return {}
        baseline = self._runner.establish_baseline()
        self._store.save_baseline(baseline)
        return {"baseline": baseline}


class ProposeNode(ResearchNode):
    name = "propose"

    def __init__(self, proposer: HypothesisProposer) -> None:
        self._proposer = proposer

    def execute(self, view: StateView) -> dict[str, Any]:
        hypothesis = self._proposer.propose(view.trace)
        logger.info("[loop %d] hypothesis: %s", view.loop, hypothesis.statement)
        return {"hypothesis": hypothesis}


class DesignNode(ResearchNode):
    name = "design"

    def __init__(self, designer: ExperimentDesigner) -> None:
        self._designer = designer

    def execute(self, view: StateView) -> dict[str, Any]:
        experiment = self._designer.design(view.require_hypothesis(), view.trace)
        logger.info("[loop %d] tasks: %s", view.loop, [task.name for task in experiment.tasks])
        return {"experiment": experiment}


class DevelopNode(ResearchNode):
    name = "develop"

    def __init__(self, developer: Developer) -> None:
        self._developer = developer

    def execute(self, view: StateView) -> dict[str, Any]:
        return {"experiment": self._developer.develop(view.require_experiment(), view.trace)}


class RunNode(ResearchNode):
    name = "run"

    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    def execute(self, view: StateView) -> dict[str, Any]:
        experiment = self._runner.run(view.require_experiment(), view.trace)
        if experiment.result is not None:
            logger.info("[loop %d] result:\n%s", view.loop, experiment.result.render())
        return {"experiment": experiment}


class SummarizeNode(ResearchNode):
    """Always yields feedback: from the summarizer, or from the failure of an earlier step."""

    name = "summarize"

    def __init__(self, summarizer: Summarizer) -> None:
        self._summarizer = summarizer

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        failure = state.get("failure")
        if failure is not None:
            return {"feedback": ExperimentFeedback.from_failure(failure)}
        update = super().__call__(state)
        if "failure" in update:
            return {"feedback": ExperimentFeedback.from_failure(update["failure"]), "failure": update["failure"]}
        return update

    def execute(self, view: StateView) -> dict[str, Any]:
        feedback = self._summarizer.summarize(view.require_experiment(), view.trace)
        logger.info("[loop %d] decision: %s — %s", view.loop, feedback.decision, feedback.reason)
        return {"feedback": feedback}


class RecordNode(ResearchNode):
    """Appends the loop to the history, persists it and clears the per-loop fields."""

    name = "record"
    recoverable = False

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def __call__(self, state: ResearchState) -> dict[str, Any]:
        feedback = state.get("feedback")
        if feedback is None:
            raise RuntimeError("The record step needs feedback")
        record = TraceRecord(
            loop=state["loop"],
            hypothesis=state.get("hypothesis"),
            experiment=state.get("experiment"),
            feedback=feedback,
        )
        self._store.save_record(record)
        return {
            "history": [record],
            "loop": state["loop"] + 1,
            "hypothesis": None,
            "experiment": None,
            "feedback": None,
            "failure": None,
        }

    def execute(self, view: StateView) -> dict[str, Any]:  # pragma: no cover - __call__ is overridden
        raise NotImplementedError


class StepRoute:
    """Conditional edge: go to ``next_step``, or straight to summarize after a failure."""

    def __init__(self, next_step: str) -> None:
        self._next = next_step

    def __call__(self, state: ResearchState) -> str:
        return SummarizeNode.name if state.get("failure") is not None else self._next


class LoopRoute:
    """Conditional edge after record: start another loop or end the run."""

    def __init__(self, settings: WorkflowSettings) -> None:
        self._settings = settings

    def __call__(self, state: ResearchState) -> str:
        if state["loop"] >= state["stop_at"]:
            return END
        failures = Trace(state["history"]).consecutive_failures()
        if failures >= self._settings.max_consecutive_failures:
            logger.error("Stopping the run after %d consecutive failed loops", failures)
            return END
        return ProposeNode.name
