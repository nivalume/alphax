"""Scenarios: the pluggable unit of the platform.

A scenario prepares a research domain and wires the five research roles for it
(:class:`ResearchComponents`). What each role's agent is told about the domain is up to that
agent's own prompt. The workflow only talks to the roles, so replacing the scenario replaces
the whole research domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel

from alphax_agent.core.experiment import Experiment
from alphax_agent.core.roles import Developer, ExperimentDesigner, HypothesisProposer, Runner, Summarizer

if TYPE_CHECKING:
    from alphax_agent.config.settings import AlphaXSettings
    from alphax_agent.llm.client import LLMClient


@dataclass(frozen=True)
class ResearchComponents:
    proposer: HypothesisProposer
    designer: ExperimentDesigner
    developer: Developer
    runner: Runner
    summarizer: Summarizer


class Scenario(ABC):
    """Base class of research scenarios.

    Subclasses set ``name`` (registry key), ``settings_model`` (validates the scenario
    options of the run file) and ``experiment_model`` (restores saved experiments with
    their concrete task type); ``headline_metrics`` are the metrics run reports show first.
    """

    name: ClassVar[str]
    title: ClassVar[str] = ""
    settings_model: ClassVar[type[BaseModel]]
    experiment_model: ClassVar[type[Experiment]] = Experiment
    headline_metrics: ClassVar[tuple[str, ...]] = ()

    def __init__(self, settings: BaseModel) -> None:
        self._settings = settings

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> Self:
        return cls(cls.settings_model.model_validate(dict(options)))

    @property
    def settings(self) -> BaseModel:
        return self._settings

    @abstractmethod
    def prepare(self, *, force: bool = False) -> None:
        """Check the environment and build derived data; idempotent unless ``force``."""

    @abstractmethod
    def assemble(self, llm: LLMClient, platform: AlphaXSettings, run_dir: Path) -> ResearchComponents:
        """Create the research roles of this scenario for one run."""
