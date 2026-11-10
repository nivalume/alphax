"""Example-only Binance download and CSV cache writer; never imported by alphax."""

from __future__ import annotations

from alphax import InstrumentId

import csv
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


_HOST = "https://api.binance.com"
_HEADER = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
    "1w": 604_800_000,
}


@dataclass(frozen=True)
class Kline:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int


class _HttpTransport:
    def __init__(self, host: str):
        self.host = host.rstrip("/")

    def get(self, url: str) -> Any:
        with urlopen(url, timeout=30) as response:
            return json.loads(response.read())

    def server_time(self) -> int:
        payload = self.get(f"{self.host}/api/v3/time")
        try:
            return int(payload["serverTime"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Binance server-time response: {payload!r}") from error

    @staticmethod
    def pace(seconds: float) -> None:
        time.sleep(seconds)


def _interval_ms(interval: str) -> int:
    try:
        return _INTERVAL_MS[interval]
    except KeyError as error:
        supported = ", ".join(_INTERVAL_MS)
        raise ValueError(
            f"unsupported interval {interval!r}; expected one of {supported}"
        ) from error


def bars_per_year(interval: str) -> float:
    """Calendar-year bars for annualizing always-open crypto markets."""
    return 365.0 * 86_400_000.0 / _interval_ms(interval)


def _milliseconds(value: Any, name: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{name} must be milliseconds or an ISO date/time") from error
    elif isinstance(value, date) and not isinstance(value, datetime):
        value = datetime.combine(value, datetime_time(), timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return int(value.timestamp() * 1000)
    raise TypeError(f"{name} must be milliseconds, an ISO string, date, or datetime")


def _payload(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid Binance JSON at byte {error.pos}") from error
    return value


def _parse_page(value: Any) -> list[Kline]:
    rows = _payload(value)
    if isinstance(rows, dict):
        raise ValueError(f"Binance returned an error: {rows!r}")
    if not isinstance(rows, list):
        raise ValueError(f"Binance klines response must be an array, got {type(rows).__name__}")

    bars = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) < 12:
            length = len(row) if isinstance(row, list) else type(row).__name__
            raise ValueError(f"kline row {index} has {length} fields, expected at least 12")
        try:
            bars.append(
                Kline(
                    open_time=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    close_time=int(row[6]),
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"kline row {index} contains a non-numeric field") from error
    return bars


def fetch_klines(
    symbol: str,
    interval: str,
    start: Any,
    *,
    end: Any = None,
    limit: int = 1000,
    pace: float = 0.25,
    closed_only: bool = True,
    max_requests: int = 1000,
    host: str = _HOST,
    transport: Any = None,
) -> list[Kline]:
    """Fetches Binance Spot klines with paging and open-bar filtering."""
    _interval_ms(interval)
    if not symbol:
        raise ValueError("symbol cannot be empty")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if max_requests < 1:
        raise ValueError("max_requests must be positive")
    if pace < 0:
        raise ValueError("pace cannot be negative")

    transport = transport or _HttpTransport(host)
    from_ms = _milliseconds(start, "start")
    end_ms = None if end is None else _milliseconds(end, "end")
    now = transport.server_time() if closed_only else None
    bars: list[Kline] = []
    requests = 0

    while True:
        if requests >= max_requests:
            raise ValueError(
                f"stopped after {requests} requests for {symbol}; raise max_requests if needed"
            )
        if requests:
            transport.pace(pace)
        query = {
            "symbol": symbol,
            "interval": interval,
            "startTime": from_ms,
            "limit": limit,
        }
        if end_ms is not None:
            query["endTime"] = end_ms
        page = _parse_page(transport.get(f"{host.rstrip('/')}/api/v3/klines?{urlencode(query)}"))
        requests += 1
        if not page:
            break

        last_open = page[-1].open_time
        if last_open < from_ms:
            raise ValueError(
                f"{symbol} returned bars ending at {last_open}, before requested {from_ms}"
            )
        bars.extend(page)
        from_ms = last_open + 1
        if len(page) < limit or (end_ms is not None and from_ms >= end_ms):
            break

    if now is not None:
        while bars and bars[-1].close_time >= now:
            bars.pop()
    return bars


def _cache_path(directory: Any, symbol: str, interval: str) -> Path:
    _interval_ms(interval)
    return Path(directory) / f"{symbol}-{interval}.csv"


def load_cached(symbol: str, interval: str, *, directory: Any = "data/binance") -> list[Kline]:
    """Loads one strict, ascending CSV cache; a missing file returns an empty list."""
    path = _cache_path(directory, symbol, interval)
    if not path.exists():
        return []
    with path.open(newline="") as file:
        rows = [row for row in csv.reader(file) if row]
    if not rows:
        return []
    if rows[0] != _HEADER:
        raise ValueError(f"{path} has header {rows[0]!r}, expected {_HEADER!r}")

    bars = []
    for line, row in enumerate(rows[1:], 2):
        if len(row) != 7:
            raise ValueError(f"{path} line {line}: {len(row)} fields, expected 7")
        try:
            bar = Kline(
                open_time=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                close_time=int(row[6]),
            )
        except ValueError as error:
            raise ValueError(f"{path} line {line} contains a non-numeric field") from error
        if bars and bar.open_time <= bars[-1].open_time:
            raise ValueError(
                f"{path} line {line}: open time {bar.open_time} does not follow "
                f"{bars[-1].open_time}"
            )
        bars.append(bar)
    return bars


def _save_cached(path: Path, bars: Sequence[Kline]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(_HEADER)
        for bar in bars:
            writer.writerow(
                [
                    bar.open_time,
                    repr(float(bar.open)),
                    repr(float(bar.high)),
                    repr(float(bar.low)),
                    repr(float(bar.close)),
                    repr(float(bar.volume)),
                    bar.close_time,
                ]
            )


def _merge(cached: list[Kline], fetched: list[Kline]) -> list[Kline]:
    if not fetched:
        return cached
    first = fetched[0].open_time
    return [bar for bar in cached if bar.open_time < first] + fetched


def fetch_cached(
    symbol: str,
    interval: str,
    start: Any,
    *,
    directory: Any = "data/binance",
    end: Any = None,
    transport: Any = None,
    **kwargs: Any,
) -> list[Kline]:
    """Resumes one symbol after its last cached bar and updates the CSV."""
    instrument = InstrumentId.parse(symbol)
    if instrument.venue != "BINANCE":
        raise ValueError("this downloader only supports venue BINANCE")
    cached = load_cached(symbol, interval, directory=directory)
    from_ms = cached[-1].open_time + _interval_ms(interval) if cached else start
    fetched = fetch_klines(
        instrument.symbol,
        interval,
        from_ms,
        end=end,
        transport=transport,
        **kwargs,
    )
    merged = _merge(cached, fetched)
    _save_cached(_cache_path(directory, symbol, interval), merged)
    return merged

