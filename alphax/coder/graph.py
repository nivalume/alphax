"""The CoSTEER search loop as a LangGraph graph.

One round::

    schedule ──Send(k)──▶ attempt ──▶ integrate ──▶ schedule ... ──▶ END

``schedule`` asks the search policy for one expansion per unsolved task, ``attempt`` runs
in parallel for every expansion (code, then evaluate), ``integrate`` adds the new nodes to
their trees and lets the reflection agent turn what changed into lessons that the next
round's attempts, on every task, can use.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel

from alphax.coder.agents import CoderAgent, ImplementationRequest, ReflectionAgent, Transition
from alphax.coder.evaluation import EvaluationPipeline
from alphax.coder.lessons import ErrorSignature, LessonBook
from alphax.coder.policy import SearchPolicy
from alphax.coder.tree import Expansion, SolutionForest, SolutionNode, SolutionTree
from alphax.config.settings import CoSTEERSettings
from alphax.core.exceptions import AgentError
from alphax.core.experiment import Experiment
from alphax.core.feedback import ImplementationFeedback, StageResult
from alphax.core.workspace import WorkspaceFactory

logger = logging.getLogger(__name__)


class Attempt(BaseModel):
    expansion: Expansion
    node: SolutionNode


class SearchSession:
    """Mutable data of one CoSTEER run: the solution trees and the shared lesson book."""

    def __init__(self, experiment: Experiment, lessons: LessonBook, root: Path) -> None:
        self.experiment = experiment
        self.forest = SolutionForest(experiment.tasks)
        self.lessons = lessons
        self.root = Path(root)
        self.task_names = {task.id: task.name for task in experiment.tasks}

    def save_summary(self) -> Path:
        """Write the explored trees (without code) for audit."""
        summary = {
            tree.task.name: [
                {
                    "id": node.id,
                    "parent": node.parent_id,
                    "operator": node.operator.value,
                    "round": node.round,
                    "score": round(node.score, 4),
                    "accepted": node.accepted,
                    "failed_stage": failure.stage if (failure := node.feedback.first_failure()) else None,
                    "workspace": str(node.workspace.path),
                }
                for node in tree.nodes
            ]
            for tree in self.forest
        }
        path = self.root / "search_tree.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return path


class AttemptLog:
    """Reducer of the ``attempts`` channel: parallel attempts accumulate, ``None`` clears."""

    @staticmethod
    def merge(current: list[Attempt], update: list[Attempt] | None) -> list[Attempt]:
        return [] if update is None else [*current, *update]


class CoSTEERState(TypedDict):
    session: SearchSession
    round: int
    plan: list[Expansion]
    attempts: Annotated[list[Attempt], AttemptLog.merge]


class AttemptInput(TypedDict):
    session: SearchSession
    expansion: Expansion
    round: int


class CoSTEERGraph:
    PEER_LIMIT = 2
    REFLECTION_BATCH = 6
    LESSON_CONTEXT = 30

    def __init__(
        self,
        coder: CoderAgent,
        pipeline: EvaluationPipeline,
        workspaces: WorkspaceFactory,
        policy: SearchPolicy,
        settings: CoSTEERSettings,
        reflector: ReflectionAgent | None = None,
    ) -> None:
        self._coder = coder
        self._pipeline = pipeline
        self._workspaces = workspaces
        self._policy = policy
        self._settings = settings
        self._reflector = reflector
        self._graph = self._build()

    def run(self, session: SearchSession) -> None:
        self._graph.invoke(
            {"session": session, "round": 0, "plan": [], "attempts": []},
            config={
                "recursion_limit": 3 * self._settings.max_rounds + 10,
                "max_concurrency": self._settings.max_parallel,
            },
        )

    def _build(self) -> Any:
        builder = StateGraph(CoSTEERState)
        builder.add_node("schedule", self._schedule)
        builder.add_node("attempt", self._attempt)  # type: ignore[arg-type]  # receives a Send payload
        builder.add_node("integrate", self._integrate)
        builder.add_edge(START, "schedule")
        builder.add_conditional_edges("schedule", self._dispatch, ["attempt", END])
        builder.add_edge("attempt", "integrate")
        builder.add_edge("integrate", "schedule")
        return builder.compile()

    def _schedule(self, state: CoSTEERState) -> dict[str, Any]:
        session = state["session"]
        if state["round"] >= self._settings.max_rounds:
            return {"plan": []}
        plan = [expansion for tree in session.forest if (expansion := self._policy.next_expansion(tree)) is not None]
        logger.info(
            "CoSTEER round %d: %d expansion(s) %s",
            state["round"],
            len(plan),
            [f"{session.task_names[e.task_id]}:{e.operator}" for e in plan],
        )
        return {"plan": plan}

    @staticmethod
    def _dispatch(state: CoSTEERState) -> list[Send] | str:
        if not state["plan"]:
            return END
        return [
            Send("attempt", {"session": state["session"], "expansion": expansion, "round": state["round"]})
            for expansion in state["plan"]
        ]

    def _attempt(self, payload: AttemptInput) -> dict[str, Any]:
        session, expansion, round_index = payload["session"], payload["expansion"], payload["round"]
        tree = session.forest.tree(expansion.task_id)
        parent = tree.get(expansion.parent_id) if expansion.parent_id is not None else None
        node_id = uuid4().hex[:10]
        workspace = self._workspaces.create(
            tree.task, self._node_path(session, tree.task.name, round_index, expansion, node_id)
        )
        if parent is not None:
            workspace.files.update(parent.files)
        request = ImplementationRequest(
            task=tree.task,
            operator=expansion.operator,
            parent=parent,
            lineage=self._lineage(tree, parent),
            lessons=session.lessons.select(
                ErrorSignature.from_feedback(parent.feedback if parent else None), self._settings.lesson_limit
            ),
            peers=session.forest.accepted_peers(tree.task.id, self.PEER_LIMIT),
            peer_names=session.task_names,
        )
        rationale = ""
        try:
            solution = self._coder.implement(request)
        except AgentError as exc:
            feedback = ImplementationFeedback(
                stages=[StageResult(stage="coding", passed=False, score=0.0, message=str(exc))]
            )
        else:
            rationale = solution.rationale
            workspace.files.update(solution.files)
            workspace.materialize()
            feedback = self._pipeline.evaluate(tree.task, workspace)
        node = SolutionNode(
            id=node_id,
            task_id=tree.task.id,
            parent_id=parent.id if parent is not None else None,
            operator=expansion.operator,
            depth=parent.depth + 1 if parent is not None else 0,
            round=round_index,
            rationale=rationale,
            workspace=workspace,
            feedback=feedback,
        )
        logger.info(
            "Attempt %s on %s: score %.2f, %s",
            expansion.operator,
            tree.task.name,
            node.score,
            "accepted" if node.accepted else f"rejected at {getattr(feedback.first_failure(), 'stage', '?')}",
        )
        return {"attempts": [Attempt(expansion=expansion, node=node)]}

    def _integrate(self, state: CoSTEERState) -> dict[str, Any]:
        session = state["session"]
        transitions: list[Transition] = []
        for attempt in state["attempts"]:
            tree = session.forest.tree(attempt.node.task_id)
            tree.add(attempt.node)
            transition = Transition.observe(tree, attempt.node)
            if transition is not None:
                transitions.append(transition)
        self._reflect(session, transitions, state["round"])
        logger.info("CoSTEER after round %d: %s", state["round"], session.forest.summary())
        return {"round": state["round"] + 1, "attempts": None, "plan": []}

    def _reflect(self, session: SearchSession, transitions: list[Transition], round_index: int) -> None:
        if self._reflector is None or self._settings.lesson_limit == 0 or not transitions:
            return
        batch = sorted(transitions, key=lambda transition: Transition.PRIORITY[transition.kind])
        batch = batch[: self.REFLECTION_BATCH]
        try:
            output = self._reflector.reflect(batch, session.lessons.select(None, self.LESSON_CONTEXT))
        except AgentError as exc:
            logger.warning("Reflection skipped in round %d: %s", round_index, exc)
            return
        origin = {transition.signature: transition.task_name for transition in reversed(batch)}
        for draft in output.lessons:
            session.lessons.add(draft, origin_task=origin.get(draft.signature, ""), round_index=round_index)
        session.lessons.reinforce(output.reinforced)

    def _lineage(self, tree: SolutionTree, parent: SolutionNode | None) -> list[SolutionNode]:
        """Ancestors of the parent shown to the coder; the parent itself is shown in full."""
        window = self._settings.lineage_window - 1
        if parent is None or window <= 0:
            return []
        return tree.ancestors(parent)[-window:]

    @staticmethod
    def _node_path(
        session: SearchSession, task_name: str, round_index: int, expansion: Expansion, node_id: str
    ) -> Path:
        safe_name = re.sub(r"[^\w.-]+", "_", task_name) or "task"
        return session.root / safe_name / f"r{round_index:02d}_{expansion.operator.value}_{node_id}"
