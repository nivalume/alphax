"""Tasks: the unit of work a developer implements."""

from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, Field


class Task(BaseModel):
    """One implementable item of an experiment, e.g. a single factor or model."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str = Field(description="Short identifier of the task")
    description: str = Field(description="What has to be implemented and why")

    def render(self) -> str:
        """Plain-text description used in prompts."""
        return f"Task: {self.name}\nDescription: {self.description}"
