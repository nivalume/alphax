"""Options of the crypto strategy scenario."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from alphax_agent.core.promotion import MetricCriterion, MetricGuard, PromotionRule
from alphax_agent.core.scenario import EnvironmentName, ScenarioSettings
from alphax_agent.env.alphax import AlphaXEnvSettings


class StrategyPeriod(BaseModel):
    """The backtest range: in-sample ``[start, oos_start)``, out-of-sample ``[oos_start, end)``.

    Only the in-sample result is shown to the agents and decides promotion; the out-of-sample result is
    recorded beside it for whoever reads the run.
    """

    start: date = date(2021, 1, 1)
    oos_start: date = date(2025, 1, 1)
    end: date = date(2026, 11, 1)

    @model_validator(mode="after")
    def _ordered(self) -> StrategyPeriod:
        if not self.start < self.oos_start < self.end:
            raise ValueError(
                f"the period needs start < oos_start < end, not {self.start}, {self.oos_start}, {self.end}"
            )
        return self

    @property
    def in_sample(self) -> tuple[date, date]:
        return self.start, self.oos_start

    @property
    def out_of_sample(self) -> tuple[date, date]:
        return self.oos_start, self.end


class UniverseSettings(BaseModel):
    """Which instruments are traded.

    Without ``symbols``, the instruments with at least ``min_bars`` in-sample bars are ranked by their median
    in-sample traded value and the first ``max_instruments`` are kept: nothing after ``oos_start`` decides it.
    """

    symbols: list[str] | None = Field(
        default=None, description="Instrument ids such as BTCUSDT-PERP.BINANCE; default: chosen by liquidity"
    )
    max_instruments: int = Field(default=30, ge=1)
    min_bars: int = Field(default=365, ge=1, description="In-sample bars an instrument needs to be chosen")

    @field_validator("symbols")
    @classmethod
    def _not_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("symbols must name at least one instrument, or be left out")
        return value


class StrategyBacktestSettings(BaseModel):
    init_cash: float = Field(default=10_000.0, gt=0)
    fees: float = Field(default=0.0005, ge=0, description="Fraction of the traded notional")
    slippage: float = Field(default=0.0002, ge=0, description="Fraction of the price, against the order")
    signal_delay: int = Field(
        default=1, ge=1, description="Bars between the bar a position is decided on and the close it trades at"
    )
    warmup_bars: int = Field(
        default=180, ge=0, description="Bars loaded before the period starts, so windows are filled when it does"
    )


class DebugSampleSettings(BaseModel):
    """The small in-sample panel the coder iterates on."""

    instruments: int = Field(default=5, ge=2)
    bars: int = Field(default=500, ge=50)
    cut_fraction: float = Field(
        default=0.75, gt=0.1, lt=0.95, description="Where the look-ahead check truncates the sample"
    )


class CryptoStrategySettings(ScenarioSettings):
    """The data (``alphax``), the backtest range (``period``) and how a strategy is backtested and judged."""

    environment: EnvironmentName = "alphax"
    alphax: AlphaXEnvSettings = Field(default_factory=AlphaXEnvSettings)
    period: StrategyPeriod = Field(default_factory=StrategyPeriod)
    universe: UniverseSettings = Field(default_factory=UniverseSettings)
    backtest: StrategyBacktestSettings = Field(default_factory=StrategyBacktestSettings)
    debug: DebugSampleSettings = Field(default_factory=DebugSampleSettings)
    cache_dir: Path = Field(
        default=Path("~/.alphax/agent/data/crypto_strategy"), description="Where the exported panels are kept"
    )
    strategy_timeout: float = Field(default=600.0, gt=0, description="Limit of one strategy computation, seconds")
    max_strategies_per_experiment: int = Field(
        default=2, ge=1, description="One mechanism and at most this many variants of it per experiment"
    )
    min_active_share: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Share of the bars a strategy must hold a position on"
    )
    promotion: PromotionRule = PromotionRule(
        primary=MetricCriterion(metric="is.portfolio.sharpe"),
        guards=[MetricGuard(metric="is.portfolio.max_drawdown", tolerance=0.05)],
    )
    use_critic: bool = Field(default=True, description="Add the LLM code review as the last evaluation stage")
    stage_weights: dict[str, float] = {"execution": 0.3, "format": 0.3, "look_ahead": 0.2, "critic": 0.2}

    @property
    def cache_path(self) -> Path:
        """``cache_dir`` with ``$VAR`` and ``~`` expanded (a literal ``~`` would point below the cwd)."""
        return Path(os.path.expandvars(str(self.cache_dir))).expanduser().resolve()
