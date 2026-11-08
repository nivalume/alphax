"""Platform settings.

Values come, in decreasing priority, from keyword overrides, ``ALPHAX_*`` environment
variables (nested with ``__``, e.g. ``ALPHAX_LLM__MODEL``), a ``.env`` file and an optional
YAML run file. Scenario options are kept as a raw mapping here and validated by the
scenario's own settings model, so the platform does not depend on any scenario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class LLMSettings(BaseModel):
    """How to reach the language model through litellm."""

    model: str = Field(default="gpt-4o-mini", description="Any litellm model id, e.g. 'deepseek/deepseek-chat'")
    temperature: float = 0.2
    max_tokens: int | None = None
    timeout: float = 180.0
    max_retries: int = 4
    retry_backoff: float = Field(default=2.0, description="Base of the exponential backoff between retries, seconds")
    api_base: str | None = None
    api_key: SecretStr | None = None
    json_mode: bool = Field(default=True, description="Ask the provider for JSON output when no tools are offered")
    cache_dir: Path | None = Field(default=None, description="Directory of the response cache; None disables it")
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Passed through to litellm.completion")

    @field_validator("cache_dir")
    @classmethod
    def _expand_cache_dir(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None


class CoSTEERSettings(BaseModel):
    """Budget and search policy of the coder."""

    policy: Literal["greedy", "ucb"] = "ucb"
    max_rounds: int = Field(default=5, ge=1, description="Search rounds; every unsolved task is expanded once a round")
    max_attempts_per_task: int = Field(default=6, ge=1)
    num_drafts: int = Field(default=1, ge=1, description="Independent first implementations per task")
    exploration: float = Field(default=0.5, ge=0.0, description="UCB exploration weight")
    max_repair_depth: int = Field(default=3, ge=1, description="Consecutive failed repairs before a branch is dropped")
    target_score: float = Field(default=0.95, ge=0.0, le=1.0, description="Stop refining a task once reached")
    lineage_window: int = Field(default=3, ge=1, description="Ancestor attempts shown to the coder")
    lesson_limit: int = Field(default=8, ge=0, description="Lessons shown to the coder per attempt")
    lesson_store: Path | None = Field(default=None, description="JSON file that keeps lessons across runs")
    max_parallel: int = Field(default=4, ge=1, description="Tasks implemented concurrently")

    @field_validator("lesson_store")
    @classmethod
    def _expand_lesson_store(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None


class WorkflowSettings(BaseModel):
    """The outer research loop."""

    max_loops: int = Field(default=10, ge=1)
    max_consecutive_failures: int = Field(default=3, ge=1, description="Stop the run after this many failed loops")
    history_window: int = Field(default=8, ge=1, description="Past loops shown to the agents")
    run_root: Path = Path("alphax_runs")
    run_name: str | None = None

    @field_validator("run_root")
    @classmethod
    def _expand_run_root(cls, value: Path) -> Path:
        return value.expanduser()


class ScenarioSelection(BaseModel):
    """Which scenario to run and its options (validated by the scenario)."""

    name: str = "qlib_factor"
    options: dict[str, Any] = Field(default_factory=dict)


class AlphaXSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ALPHAX_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    llm: LLMSettings = Field(default_factory=LLMSettings)
    costeer: CoSTEERSettings = Field(default_factory=CoSTEERSettings)
    workflow: WorkflowSettings = Field(default_factory=WorkflowSettings)
    scenario: ScenarioSelection = Field(default_factory=ScenarioSelection)

    @classmethod
    def load(cls, path: Path | str | None = None, **overrides: Any) -> AlphaXSettings:
        """Build settings, reading ``path`` (YAML) below the environment in priority."""
        if path is None:
            return cls(**overrides)
        bound = type(
            cls.__name__, (cls,), {"model_config": SettingsConfigDict(**{**cls.model_config, "yaml_file": str(path)})}
        )
        return bound(**overrides)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: tuple[PydanticBaseSettingsSource, ...] = (init_settings, env_settings, dotenv_settings)
        yaml_file = cls.model_config.get("yaml_file")
        if yaml_file:
            sources += (YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),)
        return sources

    def snapshot(self) -> dict[str, Any]:
        """Settings as plain data with secrets masked, for the run record."""
        return self.model_dump(mode="json")
