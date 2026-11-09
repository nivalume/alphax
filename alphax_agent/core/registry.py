"""Discovery of scenarios: built-ins, explicit registration and package entry points."""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import TypeVar

from alphax_agent.core.exceptions import ConfigurationError
from alphax_agent.core.scenario import Scenario

logger = logging.getLogger(__name__)

ScenarioT = TypeVar("ScenarioT", bound=type[Scenario])


class ScenarioRegistry:
    """Maps scenario names to scenario classes.

    Third-party packages add scenarios without touching this code base by declaring an
    entry point in the ``alphax_agent.scenarios`` group::

        [project.entry-points."alphax_agent.scenarios"]
        my_scenario = "my_package.scenario:MyScenario"
    """

    ENTRY_POINT_GROUP = "alphax_agent.scenarios"
    LEGACY_ENTRY_POINT_GROUP = "alphax.scenarios"
    BUILTIN_PACKAGE = "alphax_agent.scenarios"

    def __init__(self) -> None:
        self._scenarios: dict[str, type[Scenario]] = {}
        self._discovered = False

    def register(self, scenario: ScenarioT) -> ScenarioT:
        """Register a scenario class; usable as a class decorator."""
        existing = self._scenarios.get(scenario.name)
        if existing is not None and existing is not scenario:
            raise ConfigurationError(f"Scenario name {scenario.name!r} is already used by {existing.__qualname__}")
        self._scenarios[scenario.name] = scenario
        return scenario

    def get(self, name: str) -> type[Scenario]:
        self._discover()
        try:
            return self._scenarios[name]
        except KeyError:
            raise ConfigurationError(f"Unknown scenario {name!r}; available: {self.names()}") from None

    def names(self) -> list[str]:
        self._discover()
        return sorted(self._scenarios)

    def _discover(self) -> None:
        if self._discovered:
            return
        self._discovered = True
        importlib.import_module(self.BUILTIN_PACKAGE)
        for group in (self.ENTRY_POINT_GROUP, self.LEGACY_ENTRY_POINT_GROUP):
            for entry in entry_points(group=group):
                try:
                    scenario = entry.load()
                except Exception:
                    logger.exception("Could not load scenario entry point %s", entry.value)
                    continue
                if self._scenarios.get(entry.name) is not scenario:
                    self.register(scenario)


scenario_registry = ScenarioRegistry()
