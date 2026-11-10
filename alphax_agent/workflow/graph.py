"""The research loop as a LangGraph graph.

::

    START → baseline → propose → design → develop → run → summarize → record ─┐
                          ▲   (any failure jumps to summarize)                 │
                          └──────────────── next loop ◀────────────────────────┘
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import closing
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from alphax_agent.config.settings import WorkflowSettings
from alphax_agent.core.experiment import Experiment, ExperimentResult, Hypothesis, Implementation
from alphax_agent.core.feedback import ExperimentFeedback, ImplementationFeedback, StageResult, StepFailure
from alphax_agent.core.scenario import ResearchComponents
from alphax_agent.core.task import Task
from alphax_agent.core.trace import Trace, TraceRecord
from alphax_agent.core.workspace import Workspace
from alphax_agent.workflow.artifacts import ArtifactStore
from alphax_agent.workflow.nodes import (
    BaselineNode,
    DesignNode,
    DevelopNode,
    LoopRoute,
    ProposeNode,
    RecordNode,
    RunNode,
    StepRoute,
    SummarizeNode,
    TrackedStep,
)
from alphax_agent.workflow.state import ResearchState, StateView


class CheckpointSerde:
    """Serializer for LangGraph checkpointers that allows the models stored in the research state."""

    CORE_MODELS: tuple[type[BaseModel], ...] = (
        TraceRecord,
        Hypothesis,
        Experiment,
        ExperimentResult,
        Implementation,
        ExperimentFeedback,
        ImplementationFeedback,
        StageResult,
        StepFailure,
        Task,
        Workspace,
    )

    @classmethod
    def create(cls, extra_models: Iterable[type[BaseModel]] = ()) -> JsonPlusSerializer:
        """``extra_models``: the scenario's subclasses, e.g. its task, experiment and workspace types."""
        models = (*cls.CORE_MODELS, *extra_models)
        return JsonPlusSerializer(allowed_msgpack_modules=[(model.__module__, model.__name__) for model in models])


class ResearchWorkflow:
    """The research graph of one run.

    ``checkpointer`` is any LangGraph checkpointer; build it with
    ``serde=CheckpointSerde.create(...)`` so the stored state round-trips without warnings.
    """

    STEPS_PER_LOOP = 7
    STAGES = (BaselineNode.name, ProposeNode.name, DesignNode.name, DevelopNode.name, RunNode.name)
    """The stages ``run(until=...)`` can stop after."""

    def __init__(
        self,
        components: ResearchComponents,
        settings: WorkflowSettings,
        store: ArtifactStore,
        *,
        checkpointer: Any | None = None,
    ) -> None:
        self._components = components
        self._settings = settings
        self._store = store
        self._checkpointer = checkpointer
        self._graph = self.build()

    @property
    def graph(self) -> Any:
        return self._graph

    def build(self) -> Any:
        components = self._components
        builder = StateGraph(ResearchState)
        nodes = [
            BaselineNode(components.runner, self._store),
            ProposeNode(components.proposer),
            DesignNode(components.designer),
            DevelopNode(components.developer),
            RunNode(components.runner),
            SummarizeNode(components.summarizer),
            RecordNode(self._store),
        ]
        for node in nodes:
            builder.add_node(node.name, TrackedStep(node, self._store))
        builder.add_edge(START, BaselineNode.name)
        builder.add_edge(BaselineNode.name, ProposeNode.name)
        for step, following in ((ProposeNode, DesignNode), (DesignNode, DevelopNode), (DevelopNode, RunNode)):
            builder.add_conditional_edges(step.name, StepRoute(following.name), [following.name, SummarizeNode.name])
        builder.add_edge(RunNode.name, SummarizeNode.name)
        builder.add_edge(SummarizeNode.name, RecordNode.name)
        builder.add_conditional_edges(RecordNode.name, LoopRoute(self._settings), [ProposeNode.name, END])
        return builder.compile(checkpointer=self._checkpointer)

    def run(
        self,
        *,
        loops: int | None = None,
        history: Sequence[TraceRecord] = (),
        baseline: ExperimentResult | None = None,
        thread_id: str | None = None,
        until: str | None = None,
    ) -> Trace:
        """Run ``loops`` more research loops after ``history`` and return the full trace.

        ``until`` (one of :attr:`STAGES`) is a debugging aid: run one loop and stop right after that
        stage, or at the first failure. The unfinished loop is not recorded; its output is only logged.
        """
        if until is not None and until not in self.STAGES:
            raise ValueError(f"Cannot stop after {until!r}; choose one of: {', '.join(self.STAGES)}")
        count = 1 if until is not None else (loops if loops is not None else self._settings.max_loops)
        initial = StateView.initial(list(history), baseline, count)
        config: dict[str, Any] = {"recursion_limit": self.STEPS_PER_LOOP * count + 10}
        if self._checkpointer is not None:
            config["configurable"] = {"thread_id": thread_id or uuid4().hex}
        if until is not None:
            return self._run_until(initial, config, until)
        final = self._graph.invoke(initial, config=config)
        return Trace(final["history"], final.get("baseline"))

    def _run_until(self, initial: ResearchState, config: dict[str, Any], until: str) -> Trace:
        """Step through the graph and stop consuming it once ``until`` has run, so later nodes never start."""
        baseline = initial["baseline"]
        with closing(self._graph.stream(initial, config=config, stream_mode="updates")) as updates:
            for update in updates:
                for node, delta in update.items():
                    baseline = (delta or {}).get("baseline", baseline)
                    if node == until or (delta or {}).get("failure") is not None:
                        return Trace(initial["history"], baseline)
        return Trace(initial["history"], baseline)
