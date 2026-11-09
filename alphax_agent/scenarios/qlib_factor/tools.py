"""Tools of the factor agents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from alphax_agent.agents.tool import Tool
from alphax_agent.scenarios.qlib_factor.data import FactorDataCatalog


class DataProfileArgs(BaseModel):
    column: str | None = Field(default=None, description="A column such as '$close'; omit for every column")
    instrument: str | None = Field(default=None, description="Restrict to one instrument code, e.g. 'SH600000'")


class DataProfileTool(Tool[DataProfileArgs]):
    name = "profile_data"
    description = (
        "Summary statistics (count, mean, std, quantiles, NaN ratio) and the last rows of the development data "
        "sample, optionally for one column and/or one instrument."
    )
    args_model = DataProfileArgs

    def __init__(self, catalog: FactorDataCatalog) -> None:
        self._catalog = catalog

    def run(self, args: DataProfileArgs) -> str:
        return self._catalog.profile(args.column, args.instrument)
