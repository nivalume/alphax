"""Prompt templates stored as YAML and rendered with Jinja2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, Template

from alphax.core.exceptions import ConfigurationError


class PromptLibrary:
    """A keyed collection of Jinja2 templates.

    Templates are loaded from one or more YAML files whose top-level keys are template names.
    Later sources override earlier ones, so a scenario can start from the generic prompts of
    a component and replace only the templates it cares about.
    """

    _jinja = Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=False)

    def __init__(self, templates: Mapping[str, str] | None = None) -> None:
        self._sources: dict[str, str] = dict(templates or {})
        self._compiled: dict[str, Template] = {}

    @classmethod
    def from_files(cls, paths: Iterable[Path]) -> PromptLibrary:
        library = cls()
        for path in paths:
            library = library.merged(cls(cls._read_yaml(Path(path).read_text(encoding="utf-8"), str(path))))
        return library

    @classmethod
    def from_package(cls, package: str, filename: str = "prompts.yaml") -> PromptLibrary:
        text = resources.files(package).joinpath(filename).read_text(encoding="utf-8")
        return cls(cls._read_yaml(text, f"{package}/{filename}"))

    def merged(self, other: PromptLibrary) -> PromptLibrary:
        """Return a new library where the templates of ``other`` take precedence."""
        return PromptLibrary({**self._sources, **other._sources})

    def has(self, key: str) -> bool:
        return key in self._sources

    def keys(self) -> list[str]:
        return sorted(self._sources)

    def render(self, key: str, **context: Any) -> str:
        if key not in self._sources:
            raise ConfigurationError(f"Prompt template {key!r} is not defined; known templates: {self.keys()}")
        template = self._compiled.get(key)
        if template is None:
            template = self._jinja.from_string(self._sources[key])
            self._compiled[key] = template
        return template.render(**context).strip()

    @staticmethod
    def _read_yaml(text: str, origin: str) -> dict[str, str]:
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
            raise ConfigurationError(f"Prompt file {origin} must map template names to strings")
        return data
