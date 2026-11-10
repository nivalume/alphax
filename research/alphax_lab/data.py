"""Bars from the alphax catalog, with the same layout and path as the RD-Agent alphax scenario."""

from __future__ import annotations

from pathlib import Path

import alphax
import numpy as np
import pyarrow as pa

FIELDS = ("open", "high", "low", "close", "volume", "vwap")
HOUR_NS = 3_600_000_000_000


def time_ns(value: str) -> int:
    return int(np.datetime64(value, "ns").astype(np.int64))


def bar_key(symbol: str, venue: str, interval: str) -> str:
    unit = {"m": "MINUTE", "h": "HOUR", "d": "DAY", "w": "WEEK"}.get(interval[-1:])
    if unit is None or not interval[:-1].isdigit() or int(interval[:-1]) < 1:
        raise ValueError(f"invalid bar interval: {interval!r}")
    return f"{symbol}.{venue}-{int(interval[:-1])}-{unit}-LAST-EXTERNAL"


def _identity(batch, context):
    del context
    return batch


def load(catalog: str, symbols: list[str], venue: str, interval: str, start: str, end: str) -> dict:
    """Bars in [start, end), keeping only timestamps where every symbol has complete OHLCV."""
    fields = [(name, pa.float64()) for name in ("open", "high", "low", "close", "volume", "quote_volume")]
    arrow_schema = pa.schema([("event_time", pa.uint64()), ("available_time", pa.uint64()), *fields])
    schema = alphax.data.DataSchema(name="bars", input=arrow_schema, output=arrow_schema, decode=_identity)
    keys = tuple(bar_key(symbol, venue, interval) for symbol in symbols)
    source = alphax.data.Source(Path(catalog).expanduser() / "data" / "bars", schema, keys)
    dataset = source.load(start=time_ns(start), end=time_ns(end))
    panel = dataset.prepare()
    quote_volume = dataset.asof(panel, field="quote_volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(panel.volume != 0, quote_volume / panel.volume, np.nan)
    raw = {
        "open": panel.open,
        "high": panel.high,
        "low": panel.low,
        "close": panel.close,
        "volume": panel.volume,
        "vwap": vwap,
    }
    names = tuple(str(item) for item in panel.instruments)
    if names != tuple(f"{symbol}.{venue}" for symbol in symbols):
        raise ValueError(f"catalog returned unexpected instrument order: {names}")
    complete = np.logical_and.reduce(
        [np.isfinite(raw[name]).all(axis=1) for name in ("open", "high", "low", "close", "volume")]
    )
    if not complete.any():
        raise ValueError(f"no complete bars for {symbols} in [{start}, {end})")
    out: dict = {name: np.asfortranarray(value[complete], dtype=np.float64) for name, value in raw.items()}
    out["time"] = np.asarray(panel.time, dtype=np.int64)[complete]
    out["symbols"] = list(symbols)
    out["dropped_bars"] = int((~complete).sum())
    return out


def synthetic(seed: int, rows: int = 800, cols: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, (rows, cols)), axis=0))
    open_ = np.vstack([close[0], close[:-1]])
    return {
        "time": np.arange(rows, dtype=np.int64) * HOUR_NS + time_ns("2020-01-01"),
        "symbols": [f"ASSET{i}" for i in range(cols)],
        "open": np.asfortranarray(open_),
        "high": np.asfortranarray(np.maximum(open_, close) * 1.005),
        "low": np.asfortranarray(np.minimum(open_, close) * 0.995),
        "close": np.asfortranarray(close),
        "volume": np.asfortranarray(rng.uniform(100, 1000, close.shape)),
        "vwap": np.asfortranarray((open_ + close) / 2),
        "dropped_bars": 0,
    }


def last_bar_time(catalog: str, symbol: str, venue: str, interval: str) -> int | None:
    """Timestamp of the newest monthly file's last bar, for forward tracking."""
    directory = Path(catalog).expanduser() / "data" / "bars" / bar_key(symbol, venue, interval)
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return None
    import pyarrow.parquet as pq

    table = pq.read_table(files[-1], columns=["event_time"])
    return int(np.asarray(table.column("event_time")).max()) if table.num_rows else None
