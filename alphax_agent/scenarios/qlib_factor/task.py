"""Factor tasks and experiments."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from alphax_agent.core.experiment import Experiment
from alphax_agent.core.task import Task


class FactorSpec(BaseModel):
    """A factor as designed by the experiment designer."""

    name: str = Field(description="Identifier of the factor: letters, digits and underscores, starting with a letter")
    description: str = Field(description="What the factor measures and the economic reason it should predict returns")
    formulation: str = Field(description="Exact mathematical definition (LaTeX allowed), including window lengths")
    variables: dict[str, str] = Field(
        default_factory=dict, description="Every symbol of the formulation and its meaning"
    )

    @field_validator("name")
    @classmethod
    def _identifier(cls, value: str) -> str:
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
            raise ValueError("factor names must be identifiers of at most 64 characters starting with a letter")
        return value


class FactorTask(Task):
    formulation: str
    variables: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: FactorSpec) -> FactorTask:
        return cls(**spec.model_dump())

    def render(self) -> str:
        variables = "\n".join(f"  - {symbol}: {meaning}" for symbol, meaning in self.variables.items())
        return (
            f"Factor name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Formulation: {self.formulation}\n"
            f"Variables:\n{variables or '  (none)'}"
        )


class FactorExperiment(Experiment):
    tasks: list[FactorTask] = Field(default_factory=list)  # type: ignore[assignment]
