"""Test doubles and synthetic data."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from alphax_agent.llm.backend import ChatBackend
from alphax_agent.llm.types import ChatRequest, ChatResponse, TokenUsage, ToolCall

Responder = Callable[[ChatRequest], "str | ChatResponse"]


class ScriptedBackend(ChatBackend):
    """Answers by agent name, from a fixed list of replies or a function of the request."""

    def __init__(self, script: dict[str, list[str | ChatResponse] | Responder] | None = None) -> None:
        self._script = dict(script or {})
        self._cursor: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self.requests: list[ChatRequest] = []

    @property
    def model(self) -> str:
        return "scripted"

    def on(self, agent: str, script: list[str | ChatResponse] | Responder) -> ScriptedBackend:
        self._script[agent] = script
        return self

    def complete(self, request: ChatRequest) -> ChatResponse:
        with self._lock:
            self.requests.append(request)
            entry = self._script.get(request.agent)
            if entry is None:
                raise AssertionError(f"No scripted reply for agent {request.agent!r}")
            if callable(entry):
                reply = entry(request)
            else:
                index = self._cursor[request.agent]
                if index >= len(entry):
                    raise AssertionError(f"Agent {request.agent!r} asked more than {len(entry)} times")
                self._cursor[request.agent] += 1
                reply = entry[index]
        if isinstance(reply, ChatResponse):
            return reply
        return ChatResponse(content=reply, usage=TokenUsage(calls=1, prompt_tokens=10, completion_tokens=5))

    def requests_of(self, agent: str) -> list[ChatRequest]:
        return [request for request in self.requests if request.agent == agent]

    @staticmethod
    def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> ChatResponse:
        return ChatResponse(content=None, tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)])

    @staticmethod
    def as_json(payload: dict) -> str:
        return json.dumps(payload)


class SyntheticMarket:
    """Daily bars of a random-walk market with a known five-day reversal effect."""

    def __init__(
        self, instruments: int = 30, start: str = "2018-01-01", end: str = "2021-12-31", seed: int = 7
    ) -> None:
        self.dates = pd.bdate_range(start, end)
        self.symbols = [f"SH60{index:04d}" for index in range(instruments)]
        rng = np.random.default_rng(seed)
        shape = (len(self.dates), instruments)
        noise = rng.normal(0.0002, 0.02, shape)
        returns = noise.copy()
        for lag in range(1, 6):
            returns[lag:] -= 0.08 * noise[:-lag]
        close = 10 * np.exp(np.cumsum(returns, axis=0))
        open_ = close * (1 + rng.normal(0, 0.004, shape))
        self.bars = {
            "open": open_,
            "close": close,
            "high": np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, shape))),
            "low": np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, shape))),
            "volume": rng.lognormal(12, 0.4, shape),
            "factor": np.ones(shape),
        }
        self.benchmark = 10 * np.exp(np.cumsum(returns.mean(axis=1)))

    def panel(self) -> pd.DataFrame:
        index = pd.MultiIndex.from_product([self.dates, self.symbols], names=["datetime", "instrument"])
        return pd.DataFrame({f"${field}": values.reshape(-1) for field, values in self.bars.items()}, index=index)

    def write_catalog(self, root: Path, debug_instruments: int = 10, cut_fraction: float = 0.75) -> Path:
        """Write the files a FactorDataCatalog expects, without qlib."""
        full = self.panel()
        debug = full[full.index.get_level_values("instrument").isin(self.symbols[:debug_instruments])]
        debug = debug.loc["2019-01-01":"2020-12-31"]
        dates = debug.index.get_level_values("datetime").unique()
        cut = dates[int(len(dates) * cut_fraction)]
        for variant, frame in (("full", full), ("debug", debug), ("debug_cut", debug.loc[:cut])):
            (root / variant).mkdir(parents=True, exist_ok=True)
            frame.to_hdf(root / variant / "daily_pv.h5", key="data", mode="w")
        meta = {
            "market": "all",
            "fields": list(full.columns),
            "full_rows": len(full),
            "full_start": str(self.dates[0].date()),
            "full_end": str(self.dates[-1].date()),
            "full_instruments": len(self.symbols),
            "debug_rows": len(debug),
            "debug_instruments": debug_instruments,
            "debug_start": str(dates[0].date()),
            "debug_end": str(dates[-1].date()),
            "cut_date": str(cut.date()),
        }
        (root / "meta.json").write_text(json.dumps(meta))
        return root

    def write_qlib(self, root: Path, market: str = "csi300", benchmark: str = "SH000300") -> Path:
        """Write a qlib binary dataset (calendar, instruments, features)."""
        (root / "calendars").mkdir(parents=True, exist_ok=True)
        (root / "instruments").mkdir(exist_ok=True)
        (root / "calendars" / "day.txt").write_text("\n".join(date.strftime("%Y-%m-%d") for date in self.dates))
        span = f"{self.dates[0]:%Y-%m-%d}\t{self.dates[-1]:%Y-%m-%d}"
        members = [f"{symbol}\t{span}" for symbol in self.symbols]
        (root / "instruments" / f"{market}.txt").write_text("\n".join(members))
        (root / "instruments" / "all.txt").write_text("\n".join([*members, f"{benchmark}\t{span}"]))
        series = {
            symbol: {field: values[:, column] for field, values in self.bars.items()}
            for column, symbol in enumerate(self.symbols)
        }
        close = self.benchmark
        series[benchmark] = {
            "open": close,
            "close": close,
            "high": close,
            "low": close,
            "volume": np.full(len(close), 1e9),
            "factor": np.ones(len(close)),
        }
        for symbol, fields in series.items():
            folder = root / "features" / symbol.lower()
            folder.mkdir(parents=True, exist_ok=True)
            fields = {
                **fields,
                "vwap": (fields["open"] + fields["close"]) / 2,
                "change": np.r_[0.0, np.diff(fields["close"]) / fields["close"][:-1]],
            }
            for field, values in fields.items():
                np.hstack([[0.0], values]).astype("<f").tofile(folder / f"{field}.day.bin")
        return root


GOOD_FACTOR = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")
close = df["$close"]
returns = close.groupby(level="instrument").pct_change()
factor = -returns.groupby(level="instrument").transform(lambda s: s.rolling(5, min_periods=5).mean())
factor.to_frame("{name}").to_hdf("result.h5", key="data")
"""

LOOK_AHEAD_FACTOR = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")
close = df["$close"]
future = close.groupby(level="instrument").shift(-1) / close - 1
future.to_frame("{name}").to_hdf("result.h5", key="data")
"""

FULL_SAMPLE_NORMALISED_FACTOR = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")
volume = df["$volume"]
by_instrument = volume.groupby(level="instrument")
z = (volume - by_instrument.transform("mean")) / by_instrument.transform("std")
z.to_frame("{name}").to_hdf("result.h5", key="data")
"""

KEY_ERROR_FACTOR = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")
factor = df["close"].pct_change()
factor.to_frame("{name}").to_hdf("result.h5", key="data")
"""

WRONG_COLUMN_FACTOR = """
import pandas as pd

df = pd.read_hdf("daily_pv.h5", key="data")
factor = df["$close"].groupby(level="instrument").pct_change()
factor.to_frame("value").to_hdf("result.h5", key="data")
"""


BAR_SLUG = "binance.vision.market.crypto.bar.raw"
_DECIMALS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
)
_DAY_NS = 86_400 * 10**9


def bar_schema(slug: str = BAR_SLUG, version: int = 1, extra: tuple[str, ...] = (), annotations: dict | None = None):
    """A jfinance schema file of a bar dataset, as the pipeline writes it."""
    columns = [
        {"name": "instrument_id", "type": "string", "nullable": False},
        {"name": "bar_type", "type": "string", "nullable": False},
        *({"name": name, "type": "decimal", "nullable": False, "precision": 38, "scale": 18} for name in _DECIMALS),
        {"name": "trade_count", "type": "int64", "nullable": False},
        *({"name": name, "type": "decimal", "nullable": True, "precision": 38, "scale": 18} for name in extra),
        *(
            {"name": name, "type": "int64", "nullable": False}
            for name in ("ts_open", "ts_close", "ts_event", "ts_init")
        ),
    ]
    payload = {
        "id": f"custom:{slug}:{version}",
        "slug": slug,
        "version": version,
        "name": "SyntheticBar",
        "description": "Synthetic daily bars.",
        "stream_fields": {"type": "object", "properties": {"bar_type": {"type": "string"}}, "required": ["bar_type"]},
        "columns": columns,
    }
    if annotations:
        payload["annotations"] = {"fields": annotations}
    return payload


def event_schema(slug: str = "binance.vision.market.crypto.funding.raw", version: int = 1) -> dict:
    """A jfinance schema file of an event dataset (funding settlements)."""
    return {
        "slug": slug,
        "version": version,
        "name": "SyntheticFunding",
        "stream_fields": {"type": "object", "required": ["instrument_id"]},
        "columns": [
            {"name": "instrument_id", "type": "string", "nullable": False},
            {"name": "funding_rate", "type": "decimal", "nullable": False, "precision": 38, "scale": 18},
            {"name": "ts_event", "type": "int64", "nullable": False},
        ],
    }


class SyntheticJFinance:
    """Daily bars of crypto perpetuals in a jfinance store, with trending prices; the last instrument lists late."""

    def __init__(self, instruments: int = 6, start: str = "2020-01-01", end: str = "2023-12-31", seed: int = 11):
        self.days = np.arange(np.datetime64(start, "D"), np.datetime64(end, "D") + 1)
        self.symbols = [f"SYN{index}USDT-PERP.BINANCE" for index in range(instruments)]
        rng = np.random.default_rng(seed)
        shape = (len(self.days), instruments)
        drift = np.repeat(rng.normal(0, 0.004, (shape[0] // 60 + 1, instruments)), 60, axis=0)[: shape[0]]
        close = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.03, shape), axis=0))
        close[: len(self.days) // 3, -1] = np.nan
        open_ = close * (1 + rng.normal(0, 0.005, shape))
        volume = rng.lognormal(10, 0.3, shape) * np.linspace(1, 3, instruments)
        self.bars = {
            "open": open_,
            "high": np.fmax(open_, close) * 1.01,
            "low": np.fmin(open_, close) * 0.99,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "taker_buy_base_volume": volume / 2,
            "taker_buy_quote_volume": volume * close / 2,
        }

    def write(self, root: Path, market: str = "crypto", slug: str = BAR_SLUG) -> Path:
        """``<root>/<market>/schemas/<slug>.v1.json`` and one Parquet file per instrument and year."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        schemas = root / market / "schemas"
        schemas.mkdir(parents=True, exist_ok=True)
        (schemas / f"{slug}.v1.json").write_text(json.dumps(bar_schema(slug)))
        opens = self.days.astype("datetime64[ns]").astype(np.int64)
        years = self.days.astype("datetime64[Y]")
        for column, symbol in enumerate(self.symbols):
            stream = f"{symbol}-1-DAY-LAST-EXTERNAL"
            folder = root / market / f"{slug}.v1" / stream
            folder.mkdir(parents=True, exist_ok=True)
            for year in np.unique(years):
                rows = (years == year) & np.isfinite(self.bars["close"][:, column])
                if not rows.any():
                    continue
                decimals = {
                    name: pa.array(values[rows, column]).cast(pa.decimal128(38, 18), safe=False)
                    for name, values in self.bars.items()
                }
                table = pa.table(
                    {
                        "instrument_id": pa.array([symbol] * int(rows.sum())),
                        "bar_type": pa.array([stream] * int(rows.sum())),
                        **decimals,
                        "trade_count": pa.array(np.full(int(rows.sum()), 1000, dtype=np.int64)),
                        "ts_open": pa.array(opens[rows]),
                        "ts_close": pa.array(opens[rows] + _DAY_NS - 1_000_000),
                        "ts_event": pa.array(opens[rows]),
                        "ts_init": pa.array(opens[rows]),
                    }
                )
                first = f"{year}-01-01".replace("-", "")
                last = f"{int(str(year)) + 1}0101"
                pq.write_table(table, folder / f"{first}T000000Z_{last}T000000Z.parquet")
        return root

    def write_catalog(self, root: Path, debug_instruments: int = 3, debug_bars: int = 300, cut: float = 0.75) -> Path:
        """Write the panels a StrategyDataCatalog expects, without alphax: 2021 in-sample, 2023 out-of-sample."""
        opens = self.days.astype("datetime64[ns]").astype(np.int64)
        fields = {**self.bars, "trade_count": np.full_like(self.bars["close"], 1000.0)}
        fields["amount"] = fields["quote_volume"]
        in_sample = np.flatnonzero(
            (self.days >= np.datetime64("2021-01-01")) & (self.days < np.datetime64("2023-01-01"))
        )
        rows = in_sample[-debug_bars:]
        n_cut = round(len(rows) * cut)
        columns = list(range(debug_instruments))

        def save(variant: str, times: np.ndarray, symbols: list[str], arrays: dict[str, np.ndarray]) -> None:
            (root / variant).mkdir(parents=True, exist_ok=True)
            np.savez(root / variant / "panel.npz", time=times, symbols=np.array(symbols), **arrays)

        def describe(times: np.ndarray, symbols: list[str]) -> dict:
            start, end = (str(np.datetime64(int(times[i]), "ns").astype("datetime64[D]")) for i in (0, -1))
            return {"symbols": symbols, "bars": len(times), "start": start, "end": end}

        debug = {name: values[np.ix_(rows, columns)] for name, values in fields.items()}
        save("full", opens, self.symbols, fields)
        save("debug", opens[rows], self.symbols[:debug_instruments], debug)
        save(
            "debug_cut", opens[rows][:n_cut], self.symbols[:debug_instruments], {k: v[:n_cut] for k, v in debug.items()}
        )
        meta = {
            "config": {},
            "interval": "1d",
            "fields": [name for name in fields if name != "amount"],
            "aliases": {"amount": "quote_volume"},
            "start": "2021-01-01",
            "oos_start": "2023-01-01",
            "end": "2024-01-01",
            "universe": {
                "rule": "the synthetic instruments",
                "instruments": [{"symbol": s, "in_sample_bars": 700, "median_traded_value": 1.0} for s in self.symbols],
            },
            "full": {**describe(opens, self.symbols), "warmup_bars": 366},
            "debug": describe(opens[rows], self.symbols[:debug_instruments]),
            "cut_index": n_cut,
            "cut_date": str(self.days[rows][n_cut - 1]),
        }
        (root / "meta.json").write_text(json.dumps(meta))
        return root


MOMENTUM_STRATEGY = """
import numpy as np

PARAMS = {"lookback": 20}


def positions(data, lookback):
    close = data["close"]
    past = np.full_like(close, np.nan)
    past[lookback:] = close[:-lookback]
    return np.sign(close / past - 1.0)
"""

ALPHAX_MOMENTUM_STRATEGY = """
import numpy as np
import alphax

PARAMS = {"fast": 10, "slow": 40}


def positions(data, fast, slow):
    f = alphax.factors(data, fast=f"Mean($close, {fast})", slow=f"Mean($close, {slow})")
    return np.where(np.isnan(f["slow"]), np.nan, np.sign(f["fast"] - f["slow"]))
"""

LOOK_AHEAD_STRATEGY = """
import numpy as np

PARAMS = {}


def positions(data):
    close = data["close"]
    following = np.vstack([close[1:], close[-1:]])
    return np.sign(following - close)
"""

UNBOUNDED_STRATEGY = MOMENTUM_STRATEGY.replace("return np.sign(", "return 3 * np.sign(")

MISSING_FUNCTION_STRATEGY = """
PARAMS = {}


def signal(data):
    return data["close"]
"""
