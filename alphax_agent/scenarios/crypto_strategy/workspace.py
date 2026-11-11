"""Workspaces that run one strategy on one panel of the catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined

from alphax_agent.core.task import Task
from alphax_agent.core.workspace import Workspace, WorkspaceFactory
from alphax_agent.env.base import Environment, EnvResult
from alphax_agent.scenarios.crypto_strategy.data import StrategyDataCatalog


@dataclass
class StrategyExecution:
    result: EnvResult
    positions: np.ndarray | None
    read_error: str | None = None
    metrics: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.result.ok and self.positions is not None


class StrategyWorkspace(Workspace):
    """``strategy.py`` (the agent's code), the harness that runs it, and a link to one panel."""

    ENTRY: ClassVar[str] = "strategy.py"
    HARNESS: ClassVar[str] = "run_strategy.py"
    POSITIONS: ClassVar[str] = "positions.npy"
    CONFIG: ClassVar[str] = "backtest_config.json"
    METRICS: ClassVar[str] = "backtest_metrics.json"

    data_file: Path

    def prepare(self) -> None:
        self.link(self.data_file, StrategyDataCatalog.FILE_NAME)
        # Written outside ``files``: the harness is the platform's, so code the agent returns cannot replace it.
        self.resolve(self.HARNESS).write_text(_harness(), encoding="utf-8")

    def with_data(self, data_file: Path, path: Path) -> StrategyWorkspace:
        """The same code at ``path``, reading another panel."""
        return self.fork(path).model_copy(update={"data_file": Path(data_file)})

    def execute(self, environment: Environment, timeout: float) -> StrategyExecution:
        """Compute the target positions."""
        return self._run(environment, timeout, ["positions"])

    def backtest(self, environment: Environment, config: dict[str, Any], timeout: float) -> StrategyExecution:
        """Compute the positions and backtest them; ``metrics`` holds the harness's metrics file."""
        self.materialize()
        self.resolve(self.CONFIG).write_text(json.dumps(config, indent=2), encoding="utf-8")
        execution = self._run(environment, timeout, ["backtest", self.CONFIG])
        if not execution.ok:
            return execution
        path = self.resolve(self.METRICS)
        if not path.exists():
            execution.read_error = f"{self.METRICS} was not written"
            execution.positions = None
            return execution
        execution.metrics = json.loads(path.read_text(encoding="utf-8"))
        return execution

    def _run(self, environment: Environment, timeout: float, args: list[str]) -> StrategyExecution:
        self.materialize()
        output = self.resolve(self.POSITIONS)
        output.unlink(missing_ok=True)
        self.resolve(self.METRICS).unlink(missing_ok=True)
        result = environment.run_python(self.HARNESS, self.path, args=args, timeout=timeout)
        result = result.model_copy(update={"output": result.output.replace(str(self.path.resolve()), "<workspace>")})
        if not result.ok:
            return StrategyExecution(result=result, positions=None)
        if not output.exists():
            return StrategyExecution(result=result, positions=None, read_error=f"{self.POSITIONS} was not written")
        try:
            positions = np.load(output, allow_pickle=False)
        except Exception as exc:
            return StrategyExecution(result=result, positions=None, read_error=f"{self.POSITIONS} is unreadable: {exc}")
        return StrategyExecution(result=result, positions=positions)


class StrategyWorkspaceFactory(WorkspaceFactory):
    """New implementations develop against the debug panel."""

    def __init__(self, catalog: StrategyDataCatalog) -> None:
        self._catalog = catalog

    def create(self, task: Task, path: Path) -> StrategyWorkspace:
        return StrategyWorkspace(path=path, data_file=self._catalog.path("debug"))


@cache
def _harness() -> str:
    source = resources.files("alphax_agent.scenarios.crypto_strategy").joinpath("templates/run_strategy.py.j2")
    return (
        JinjaEnvironment(undefined=StrictUndefined, keep_trailing_newline=True)
        .from_string(source.read_text(encoding="utf-8"))
        .render(
            entry=StrategyWorkspace.ENTRY,
            harness=StrategyWorkspace.HARNESS,
            panel=StrategyDataCatalog.FILE_NAME,
            positions=StrategyWorkspace.POSITIONS,
            config=StrategyWorkspace.CONFIG,
            metrics=StrategyWorkspace.METRICS,
        )
    )
