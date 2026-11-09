"""Search policies: which node of a solution tree to expand next, and how."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

from alphax_agent.coder.tree import Expansion, Operator, SolutionNode, SolutionTree
from alphax_agent.config.settings import CoSTEERSettings
from alphax_agent.core.exceptions import ConfigurationError


class SearchPolicy(ABC):
    """Shared budget rules; subclasses only choose among expandable nodes.

    A task stops receiving attempts when it is solved (accepted with a score at or above
    the target) or its attempt budget is spent. A rejected node is repaired only while it is
    the leaf of its branch, so a repair chain is followed rather than forked, and only while
    the chain has failed fewer than ``max_repair_depth`` times in a row. Accepted nodes below
    the target can be improved several times. If no node is expandable the policy starts a
    fresh draft instead of insisting on a dead branch.
    """

    key: ClassVar[str]
    _registry: ClassVar[dict[str, type[SearchPolicy]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "key", None):
            SearchPolicy._registry[cls.key] = cls

    def __init__(self, settings: CoSTEERSettings) -> None:
        self._settings = settings

    @classmethod
    def create(cls, settings: CoSTEERSettings) -> SearchPolicy:
        try:
            return cls._registry[settings.policy](settings)
        except KeyError:
            raise ConfigurationError(
                f"Unknown search policy {settings.policy!r}; known: {sorted(cls._registry)}"
            ) from None

    def next_expansion(self, tree: SolutionTree) -> Expansion | None:
        settings = self._settings
        if tree.attempts >= settings.max_attempts_per_task or tree.solved(settings.target_score):
            return None
        if tree.drafts < settings.num_drafts:
            return Expansion(task_id=tree.task.id, operator=Operator.DRAFT)
        candidates = [node for node in tree.nodes if self._expandable(tree, node)]
        if not candidates:
            return Expansion(task_id=tree.task.id, operator=Operator.DRAFT)
        node = self._choose(tree, candidates)
        operator = Operator.IMPROVE if node.accepted else Operator.REPAIR
        return Expansion(task_id=tree.task.id, operator=operator, parent_id=node.id)

    def _expandable(self, tree: SolutionTree, node: SolutionNode) -> bool:
        if node.accepted:
            return node.score < self._settings.target_score
        is_leaf = not tree.children(node.id)
        return is_leaf and tree.failure_streak(node) < self._settings.max_repair_depth

    @abstractmethod
    def _choose(self, tree: SolutionTree, candidates: list[SolutionNode]) -> SolutionNode: ...


class GreedyPolicy(SearchPolicy):
    """Always expand the best node: accepted first, then score, then the most recent."""

    key = "greedy"

    def _choose(self, tree: SolutionTree, candidates: list[SolutionNode]) -> SolutionNode:
        return max(candidates, key=lambda node: (node.accepted, node.score, node.round))


class UCBPolicy(SearchPolicy):
    """Upper confidence bound over the nodes of the tree.

    ``value = score + c * sqrt(ln(N) / (1 + n))`` where ``N`` counts all expansions of the
    tree and ``n`` those of the node, so a promising node that was already refined several
    times yields to a less explored sibling with a similar score.
    """

    key = "ucb"

    def _choose(self, tree: SolutionTree, candidates: list[SolutionNode]) -> SolutionNode:
        total = sum(1 + node.expansions for node in tree.nodes)
        log_total = math.log(total + 1)
        exploration = self._settings.exploration

        def value(node: SolutionNode) -> tuple[float, int]:
            bonus = exploration * math.sqrt(log_total / (1 + node.expansions))
            return node.score + bonus, node.round

        return max(candidates, key=value)
