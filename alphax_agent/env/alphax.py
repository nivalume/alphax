"""The alphax runtime: an interpreter with the alphax backtesting engine, and the jfinance data it reads."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from alphax_agent.core.exceptions import EnvironmentUnavailableError
from alphax_agent.env.base import Environment, EnvResult
from alphax_agent.env.local import LocalEnvironment

#: Accepted spellings of each supported bar interval, and the interval they mean.
_INTERVALS = {"1d": "1d", "d": "1d", "day": "1d", "1day": "1d", "daily": "1d"}


def _expand(value: str | Path) -> Path:
    """``$VAR`` and ``~`` expanded, made absolute against the current directory."""
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


class AlphaXEnvSettings(BaseModel):
    """Where alphax runs and where its data is.

    The data is a jfinance store: ``<data_dir>/<market>/schemas/*.json`` describe the datasets, and each
    dataset's rows are under ``<data_dir>/<market>/<slug>.v<version>/<stream>/*.parquet``, one stream per
    instrument and bar interval.
    """

    python_bin: str | None = Field(
        default=None,
        description="Interpreter with alphax (alphax-trader) and pyarrow installed; default: this one. "
        "A relative path is taken from the current directory.",
    )
    data_dir: Path = Field(default=Path("~/.jfinance"), description="Root of the jfinance data store")
    market: str = Field(default="crypto", description="The market directory under data_dir")
    interval: str = Field(default="1d", description="Bar interval of the data; only daily bars ('1d') for now")
    schema_dir: Path | None = Field(default=None, description="Schema files; default <data_dir>/<market>/schemas")
    schemas: list[str] = Field(
        default_factory=lambda: ["binance.vision.market.crypto.bar.raw"],
        min_length=1,
        description="Slugs of the datasets given to the agents; each must be a bar dataset with rows on disk",
    )
    timeout: float = Field(default=1800.0, gt=0, description="Limit of one data export or backtest, seconds")
    threads: int = Field(default=1, ge=1, description="Threads of one alphax backtest")

    @field_validator("interval")
    @classmethod
    def _supported_interval(cls, value: str) -> str:
        interval = _INTERVALS.get(value.strip().lower())
        if interval is None:
            raise ValueError(f"interval {value!r} is not supported yet; only daily bars are: use '1d'")
        return interval

    @field_validator("python_bin")
    @classmethod
    def _interpreter_path(cls, value: str | None) -> str | None:
        # A bare name ("python3") is looked up on PATH; a path is fixed now, because commands run in other
        # directories. Symlinks are kept: a venv's python is a link to the base interpreter, which lacks the venv.
        if value is None or os.sep not in value:
            return value
        return os.path.abspath(os.path.expanduser(os.path.expandvars(value)))

    @property
    def data_path(self) -> Path:
        return _expand(self.data_dir)

    @property
    def market_path(self) -> Path:
        return self.data_path / self.market

    @property
    def schema_path(self) -> Path:
        return _expand(self.schema_dir) if self.schema_dir is not None else self.market_path / "schemas"

    @property
    def stream_suffix(self) -> str:
        """The end of a stream directory's name, after the instrument (``BTCUSDT-PERP.BINANCE-1-DAY-LAST-EXTERNAL``)."""
        return {"1d": "1-DAY-LAST-EXTERNAL"}[self.interval]

    @property
    def bar_nanoseconds(self) -> int:
        return {"1d": 86_400 * 10**9}[self.interval]

    @property
    def ann_factor(self) -> float:
        """Bars per year: crypto trades every day of the year."""
        return {"1d": 365.0}[self.interval]


class AlphaXEnvironment(Environment):
    """alphax on top of an execution backend (a local interpreter by default)."""

    PROBE_PACKAGES = ("numpy", "pyarrow", "alphax-trader")

    def __init__(self, settings: AlphaXEnvSettings, backend: Environment | None = None) -> None:
        self._settings = settings
        self._backend = backend or LocalEnvironment(settings.python_bin, default_timeout=settings.timeout)

    @property
    def settings(self) -> AlphaXEnvSettings:
        return self._settings

    @property
    def python_command(self) -> list[str]:
        return self._backend.python_command

    def prepare(self) -> None:
        self._backend.prepare()
        probe = self._backend.run(
            [*self.python_command, "-c", "import alphax, numpy, pyarrow; print(alphax.__file__)"],
            Path.cwd(),
            timeout=120,
        )
        if not probe.ok:
            raise EnvironmentUnavailableError(
                "alphax is not importable in the execution environment; build it with `uv sync` in the alphax "
                f"repository and point the alphax.python_bin option at that .venv/bin/python:\n{probe.tail()}"
            )
        settings = self._settings
        if not settings.data_path.is_dir():
            raise EnvironmentUnavailableError(f"The data directory {settings.data_path} does not exist")
        if not settings.schema_path.is_dir():
            raise EnvironmentUnavailableError(f"The schema directory {settings.schema_path} does not exist")

    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> EnvResult:
        return self._backend.run(command, workdir, env=env, timeout=timeout)
