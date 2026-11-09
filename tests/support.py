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
