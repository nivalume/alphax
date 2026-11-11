"""Tools of the crypto strategy agents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from alphax_agent.agents.tool import Tool
from alphax_agent.scenarios.crypto_strategy.data import StrategyDataCatalog


class PanelProfileArgs(BaseModel):
    field: str | None = Field(default=None, description="A field such as 'close'; omit for every field")
    instrument: str | None = Field(default=None, description="Restrict to one instrument, e.g. 'BTCUSDT-PERP.BINANCE'")


class PanelProfileTool(Tool[PanelProfileArgs]):
    name = "profile_data"
    description = (
        "Summary statistics (mean, std, quantiles, NaN ratio) of the fields of the development panel, optionally "
        "for one field and/or one instrument."
    )
    args_model = PanelProfileArgs

    def __init__(self, catalog: StrategyDataCatalog) -> None:
        self._catalog = catalog

    def run(self, args: PanelProfileArgs) -> str:
        return self._catalog.profile(args.field, args.instrument)
