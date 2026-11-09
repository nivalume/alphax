"""Workspaces that compute one factor from the catalog data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from alphax_agent.core.task import Task
from alphax_agent.core.workspace import Workspace, WorkspaceFactory
from alphax_agent.env.base import Environment, EnvResult
from alphax_agent.scenarios.qlib_factor.data import FactorDataCatalog


@dataclass
class FactorExecution:
    result: EnvResult
    values: pd.DataFrame | None
    read_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result.ok and self.values is not None

    def first_column(self) -> pd.Series:
        """The factor values of a successful execution."""
        if self.values is None:
            raise ValueError("The execution produced no values")
        return self.values.iloc[:, 0]


class FactorWorkspace(Workspace):
    """``factor.py`` plus a link to one variant of the data; writes ``result.h5``."""

    ENTRY: ClassVar[str] = "factor.py"
    OUTPUT: ClassVar[str] = "result.h5"
    VALUES: ClassVar[str] = "factor_values.parquet"

    data_file: Path

    def prepare(self) -> None:
        self.link(self.data_file, FactorDataCatalog.FILE_NAME)

    def with_data(self, data_file: Path, path: Path) -> FactorWorkspace:
        """The same code at ``path``, reading another data file."""
        return self.fork(path).model_copy(update={"data_file": Path(data_file)})

    def execute(self, environment: Environment, timeout: float) -> FactorExecution:
        self.materialize()
        output = self.resolve(self.OUTPUT)
        output.unlink(missing_ok=True)
        result = environment.run_python(self.ENTRY, self.path, timeout=timeout)
        result = result.model_copy(update={"output": result.output.replace(str(self.path.resolve()), "<workspace>")})
        if not result.ok:
            return FactorExecution(result=result, values=None)
        if not output.exists():
            return FactorExecution(result=result, values=None, read_error=f"{self.OUTPUT} was not written")
        try:
            values = pd.read_hdf(output)
        except Exception as exc:
            return FactorExecution(result=result, values=None, read_error=f"{self.OUTPUT} is unreadable: {exc}")
        if isinstance(values, pd.Series):
            values = values.to_frame()
        return FactorExecution(result=result, values=values)


class FactorWorkspaceFactory(WorkspaceFactory):
    """New implementations develop against the debug sample."""

    def __init__(self, catalog: FactorDataCatalog) -> None:
        self._catalog = catalog

    def create(self, task: Task, path: Path) -> FactorWorkspace:
        return FactorWorkspace(path=path, data_file=self._catalog.path("debug"))
