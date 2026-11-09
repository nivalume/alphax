"""Search trees of candidate implementations.

Every task of an experiment owns one :class:`SolutionTree`. A node is one complete
implementation together with its evaluation; an edge records which operator produced the
child from its parent. Drafts are roots, repairs and improvements are descendants.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, SerializeAsAny

from alphax_agent.core.feedback import ImplementationFeedback
from alphax_agent.core.task import Task
from alphax_agent.core.workspace import Workspace


class Operator(StrEnum):
    DRAFT = "draft"
    REPAIR = "repair"
    IMPROVE = "improve"


class Expansion(BaseModel):
    """A decision of the search policy: apply ``operator`` to ``parent_id`` of a task."""

    task_id: str
    operator: Operator
    parent_id: str | None = None


class SolutionNode(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:10])
    task_id: str
    parent_id: str | None = None
    operator: Operator
    depth: int = 0
    round: int = 0
    rationale: str = ""
    workspace: SerializeAsAny[Workspace]
    feedback: ImplementationFeedback
    expansions: int = 0

    @property
    def score(self) -> float:
        return self.feedback.score

    @property
    def accepted(self) -> bool:
        return self.feedback.accepted

    @property
    def files(self) -> dict[str, str]:
        return self.workspace.files


class SolutionTree:
    """All attempts at one task."""

    def __init__(self, task: Task) -> None:
        self._task = task
        self._nodes: dict[str, SolutionNode] = {}

    @property
    def task(self) -> Task:
        return self._task

    @property
    def nodes(self) -> list[SolutionNode]:
        return list(self._nodes.values())

    @property
    def attempts(self) -> int:
        return len(self._nodes)

    @property
    def drafts(self) -> int:
        return sum(1 for node in self._nodes.values() if node.operator is Operator.DRAFT)

    def add(self, node: SolutionNode) -> None:
        if node.task_id != self._task.id:
            raise ValueError(f"Node {node.id} belongs to task {node.task_id}, not {self._task.id}")
        self._nodes[node.id] = node
        if node.parent_id is not None and node.parent_id in self._nodes:
            self._nodes[node.parent_id].expansions += 1

    def get(self, node_id: str) -> SolutionNode:
        return self._nodes[node_id]

    def children(self, node_id: str) -> list[SolutionNode]:
        return [node for node in self._nodes.values() if node.parent_id == node_id]

    def ancestors(self, node: SolutionNode) -> list[SolutionNode]:
        """Ancestors of ``node``, oldest first, excluding ``node``."""
        chain: list[SolutionNode] = []
        current = node
        while current.parent_id is not None and current.parent_id in self._nodes:
            current = self._nodes[current.parent_id]
            chain.append(current)
        return list(reversed(chain))

    def failure_streak(self, node: SolutionNode) -> int:
        """Number of consecutive rejected nodes ending at ``node`` along its branch."""
        streak = 0
        for current in [node, *reversed(self.ancestors(node))]:
            if current.accepted:
                break
            streak += 1
        return streak

    def best(self) -> SolutionNode | None:
        """Accepted nodes beat rejected ones, then higher score, then the more recent attempt."""
        if not self._nodes:
            return None
        return max(self._nodes.values(), key=lambda node: (node.accepted, node.score, node.round))

    def solved(self, target_score: float) -> bool:
        best = self.best()
        return best is not None and best.accepted and best.score >= target_score


class SolutionForest:
    """The trees of all tasks of one experiment."""

    def __init__(self, tasks: Sequence[Task]) -> None:
        self._trees = {task.id: SolutionTree(task) for task in tasks}

    def __iter__(self) -> Iterator[SolutionTree]:
        return iter(self._trees.values())

    def tree(self, task_id: str) -> SolutionTree:
        return self._trees[task_id]

    def best_nodes(self) -> dict[str, SolutionNode]:
        return {task_id: best for task_id, tree in self._trees.items() if (best := tree.best()) is not None}

    def accepted_peers(self, task_id: str, limit: int) -> list[SolutionNode]:
        """Best accepted implementations of the other tasks, shared with the coder of ``task_id``."""
        peers = [
            best
            for other_id, tree in self._trees.items()
            if other_id != task_id and (best := tree.best()) is not None and best.accepted
        ]
        return sorted(peers, key=lambda node: node.score, reverse=True)[:limit]

    def summary(self) -> str:
        lines = []
        for tree in self._trees.values():
            best = tree.best()
            status = (
                "no attempt" if best is None else f"best {best.score:.2f} {'accepted' if best.accepted else 'rejected'}"
            )
            lines.append(f"{tree.task.name}: {tree.attempts} attempts, {status}")
        return "; ".join(lines)
