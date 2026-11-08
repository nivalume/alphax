"""The research loop as a LangGraph graph.

::

    START → baseline → propose → design → develop → run → summarize → record ─┐
                          ▲   (any failure jumps to summarize)                 │
                          └──────────────── next loop ◀────────────────────────┘
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from alphax.config.settings import WorkflowSettings
from alphax.core.experiment import Experiment, ExperimentResult, Hypothesis, Implementation
from alphax.core.feedback import ExperimentFeedback, ImplementationFeedback, StageResult, StepFailure
from alphax.core.scenario import ResearchComponents
from alphax.core.task import Task
from alphax.core.trace import Trace, TraceRecord
from alphax.core.workspace import Workspace
from alphax.workflow.artifacts import ArtifactStore
from alphax.workflow.nodes import (
    BaselineNode,
    DesignNode,
    DevelopNode,
    LoopRoute,
    ProposeNode,
    RecordNode,
    RunNode,
    StepRoute,
    SummarizeNode,
)
from alphax.workflow.state import ResearchState, StateView


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
            builder.add_node(node.name, node)
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
    ) -> Trace:
        """Run ``loops`` more research loops after ``history`` and return the full trace."""
        count = loops if loops is not None else self._settings.max_loops
        initial = StateView.initial(list(history), baseline, count)
        config: dict[str, Any] = {"recursion_limit": self.STEPS_PER_LOOP * count + 10}
        if self._checkpointer is not None:
            config["configurable"] = {"thread_id": thread_id or uuid4().hex}
        final = self._graph.invoke(initial, config=config)
        return Trace(final["history"], final.get("baseline"))
