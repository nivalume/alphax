"""Hypotheses, experiments and their results."""

from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from pydantic import BaseModel, Field, SerializeAsAny

from alphax.core.feedback import ImplementationFeedback
from alphax.core.task import Task
from alphax.core.workspace import Workspace


class Hypothesis(BaseModel):
    """A falsifiable research idea that one experiment tests."""

    statement: str = Field(description="The hypothesis in one or two sentences")
    rationale: str = Field(description="Why the hypothesis should hold: mechanism, counterparty, evidence")
    observation: str = Field(default="", description="What in the data or the history motivated it")
    justification: str = Field(default="", description="Why it is the most promising next step given the history")
    knowledge: str = Field(default="", description="Reusable knowledge the experiment will confirm or reject")

    def render(self) -> str:
        lines = [f"Hypothesis: {self.statement}", f"Rationale: {self.rationale}"]
        if self.observation:
            lines.append(f"Observation: {self.observation}")
        if self.justification:
            lines.append(f"Justification: {self.justification}")
        return "\n".join(lines)


class ExperimentResult(BaseModel):
    """Measured outcome of running an experiment."""

    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list, description="Facts about how the result was obtained")
    log: str = ""

    def get(self, name: str) -> float | None:
        return self.metrics.get(name)

    def render(self, keys: Iterable[str] | None = None) -> str:
        names = list(keys) if keys is not None else sorted(self.metrics)
        rows = [f"- {name}: {self.metrics[name]:.6f}" for name in names if name in self.metrics]
        rows += [f"- note: {note}" for note in self.notes]
        return "\n".join(rows) if rows else "(no metrics)"


class Implementation(BaseModel):
    """The selected implementation of one task together with its evaluation."""

    task_id: str
    workspace: SerializeAsAny[Workspace]
    feedback: ImplementationFeedback
    artifacts: dict[str, str] = Field(default_factory=dict)


class Experiment(BaseModel):
    """A set of tasks that together test one hypothesis."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    hypothesis: Hypothesis | None = None
    tasks: list[SerializeAsAny[Task]] = Field(default_factory=list)
    implementations: dict[str, Implementation] = Field(default_factory=dict)
    workspace: SerializeAsAny[Workspace] | None = None
    result: ExperimentResult | None = None

    def task(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(task_id)

    def accepted(self) -> list[tuple[Task, Implementation]]:
        """Tasks whose selected implementation passed every evaluation stage, in task order."""
        return [
            (task, self.implementations[task.id])
            for task in self.tasks
            if task.id in self.implementations and self.implementations[task.id].feedback.accepted
        ]

    def render_tasks(self) -> str:
        return "\n\n".join(task.render() for task in self.tasks) if self.tasks else "(no tasks)"
