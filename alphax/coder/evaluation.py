"""Staged evaluation of an implementation.

Stages run in order and stop at the first failure: there is no point asking a critic about
code that does not run. Deterministic stages come first; LLM-backed stages (a code critic)
come last, so most rejections cost no tokens.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from alphax.core.exceptions import AlphaXError
from alphax.core.feedback import ImplementationFeedback, StageResult
from alphax.core.task import Task
from alphax.core.workspace import Workspace

logger = logging.getLogger(__name__)


class EvaluationContext:
    """What the stages of one evaluation share: the subject and intermediate artifacts."""

    def __init__(self, task: Task, workspace: Workspace) -> None:
        self.task = task
        self.workspace = workspace
        self._artifacts: dict[str, Any] = {}

    def put(self, name: str, value: Any) -> None:
        self._artifacts[name] = value

    def get(self, name: str) -> Any:
        if name not in self._artifacts:
            raise KeyError(f"Artifact {name!r} was not produced by an earlier stage")
        return self._artifacts[name]


class EvaluationStage(ABC):
    name: ClassVar[str]

    def __init__(self, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError("Stage weight must be non-negative")
        self.weight = weight

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> StageResult: ...

    def result(self, passed: bool, score: float, message: str) -> StageResult:
        return StageResult(stage=self.name, passed=passed, score=max(0.0, min(1.0, score)), message=message)


class EvaluationPipeline:
    def __init__(self, stages: Sequence[EvaluationStage]) -> None:
        if not stages:
            raise ValueError("An evaluation pipeline needs at least one stage")
        self._stages = list(stages)

    @property
    def stage_names(self) -> list[str]:
        return [stage.name for stage in self._stages]

    def evaluate(self, task: Task, workspace: Workspace) -> ImplementationFeedback:
        context = EvaluationContext(task, workspace)
        results: list[StageResult] = []
        blocked = False
        for stage in self._stages:
            if blocked:
                results.append(
                    StageResult(
                        stage=stage.name,
                        passed=False,
                        score=0.0,
                        skipped=True,
                        message="Skipped because an earlier stage failed.",
                    )
                )
                continue
            try:
                outcome = stage.evaluate(context)
            except AlphaXError as exc:
                logger.info("Stage %s raised %s for %s", stage.name, type(exc).__name__, task.name)
                outcome = StageResult(stage=stage.name, passed=False, score=0.0, message=f"{type(exc).__name__}: {exc}")
            results.append(outcome)
            blocked = not outcome.passed
        return ImplementationFeedback.from_stages(results, [stage.weight for stage in self._stages])
