"""Download the complete Binance USD-M futures kline history into alphax's catalog.

The default destination is ``~/.cache/alphax/catalog`` and the resulting layout is::

    data/bars/BTCUSDT.BINANCE-15-MINUTE-LAST-EXTERNAL/2020-01.parquet

Binance publishes closed months as monthly archives and the current month as
daily archives.  The script prefers monthly files, falls back to daily files
when a month is not published, and is safe to run again: completed Parquet
parts are skipped.  It downloads only public Vision files; no API key is
needed.

Examples::

    python examples/download_binance_vision.py --dry-run
    python examples/download_binance_vision.py --intervals 15m 1h 4h 1d
    python examples/download_binance_vision.py --symbols BTCUSDT ETHUSDT \
        --start 2024-01 --end 2024-03 --workers 4

The default output is normalized Parquet with the five first-class Bar fields,
event/availability timestamps in UTC nanoseconds, and the vendor fields kept
under their own names.  This is still ordinary user data: ``alphax.data`` needs
an explicit DataSchema to project it into a BarPanel.
"""

from __future__ import annotations

import argparse
import calendar
import csv
import hashlib
import io
import itertools
import json
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE_URL = "https://data.binance.vision/data/futures/um"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT")
DEFAULT_INTERVALS = ("15m", "1h", "4h", "1d")
INTERVALS = {"15m": (15, "MINUTE"), "1h": (1, "HOUR"), "4h": (4, "HOUR"), "1d": (1, "DAY")}
VISION_COLUMNS = (
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


@dataclass(frozen=True)
class Archive:
    symbol: str
    interval: str
    period: str
    url: str
    output: Path


def _month_cursor(value: str | date) -> date:
    if isinstance(value, date):
        return value.replace(day=1)
    try:
        return date.fromisoformat(f"{value}-01")
    except ValueError as error:
        raise ValueError(f"expected YYYY-MM, got {value!r}") from error


def _next_month(value: date) -> date:
    return value.replace(year=value.year + (value.month == 12), month=1 if value.month == 12 else value.month + 1)


def _iter_months(start: date, end: date) -> Iterable[date]:
    current = start
    while current < end:
        yield current
        current = _next_month(current)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"expected YYYY-MM-DD, got {value!r}") from error


def _symbol(value: str) -> str:
    value = value.strip().upper()
    if not value or any(ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for ch in value):
        raise ValueError(f"invalid Binance symbol: {value!r}")
    return value


def _bar_key(symbol: str, interval: str) -> str:
    step, unit = INTERVALS[interval]
    return f"{symbol}.BINANCE-{step}-{unit}-LAST-EXTERNAL"


def _request(url: str) -> bytes | None:
    request = Request(url, headers={"User-Agent": "alphax-binance-vision/0.1"})
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except HTTPError as error:
        if error.code == 404:
            return None
        raise RuntimeError(f"Binance Vision HTTP {error.code}: {url}") from error
    except URLError as error:
        raise RuntimeError(f"Binance Vision request failed: {url}: {error.reason}") from error


def _checksum(url: str, payload: bytes) -> str | None:
    checksum = _request(f"{url}.CHECKSUM")
    if not checksum:
        return None
    expected = checksum.decode("utf-8").split()[0].lower()
    actual = hashlib.sha256(payload).hexdigest()
    if expected != actual:
        raise RuntimeError(f"SHA-256 mismatch for {url}: expected {expected}, got {actual}")
    return actual


def _archive_url(symbol: str, interval: str, period: str, frequency: str) -> str:
    filename = f"{symbol}-{interval}-{period}.zip"
    return f"{BASE_URL}/{frequency}/klines/{symbol}/{interval}/{filename}"


def _parse_rows(payload: bytes) -> pa.Table:
    with ZipFile(io.BytesIO(payload)) as archive:
        members = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"expected one CSV in Vision archive, found {members}")
        with archive.open(members[0]) as stream:
            rows = csv.reader(io.TextIOWrapper(stream, encoding="utf-8", newline=""))
            first = next(rows, None)
            if first is None:
                raise RuntimeError(f"empty Vision archive: {members[0]}")
            if first and first[0].lower() == "open_time":
                first = None
            else:
                rows = itertools.chain((first,), rows)
            collected = [[] for _ in VISION_COLUMNS]
            previous = None
            for line, row in enumerate(rows, 2):
                if len(row) != len(VISION_COLUMNS):
                    raise RuntimeError(f"{members[0]} line {line}: expected 12 fields, got {len(row)}")
                try:
                    opened = int(row[0])
                    if previous is not None and opened <= previous:
                        raise RuntimeError(f"{members[0]} line {line}: non-increasing open time")
                    previous = opened
                    numeric = [opened, float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), int(row[6]), float(row[7]), int(row[8]), float(row[9]), float(row[10]), float(row[11])]
                except (TypeError, ValueError) as error:
                    raise RuntimeError(f"{members[0]} line {line}: invalid numeric field") from error
                for values, value in zip(collected, numeric):
                    values.append(value)

    if not collected[0]:
        raise RuntimeError(f"empty Vision CSV: {members[0]}")
    open_time = pa.array(collected[0], type=pa.uint64())
    close_time = pa.array(collected[6], type=pa.uint64())
    columns = {
        "event_time": pc.call_function("multiply", [open_time, pa.scalar(1_000_000, pa.uint64())]),
        "available_time": pc.call_function("multiply", [close_time, pa.scalar(1_000_000, pa.uint64())]),
        "open": pa.array(collected[1], type=pa.float64()),
        "high": pa.array(collected[2], type=pa.float64()),
        "low": pa.array(collected[3], type=pa.float64()),
        "close": pa.array(collected[4], type=pa.float64()),
        "volume": pa.array(collected[5], type=pa.float64()),
        "open_time_ms": open_time,
        "close_time_ms": close_time,
        "quote_volume": pa.array(collected[7], type=pa.float64()),
        "trade_count": pa.array(collected[8], type=pa.uint64()),
        "taker_buy_volume": pa.array(collected[9], type=pa.float64()),
        "taker_buy_quote_volume": pa.array(collected[10], type=pa.float64()),
        "ignore": pa.array(collected[11], type=pa.float64()),
    }
    return pa.table(columns)


def _write_part(archive: Archive, payload: bytes, *, keep_archive: bool) -> int:
    archive.output.parent.mkdir(parents=True, exist_ok=True)
    table = _parse_rows(payload)
    with NamedTemporaryFile(prefix=archive.output.name, suffix=".part", dir=archive.output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, archive.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    if keep_archive:
        raw = archive.output.with_suffix(".zip")
        raw.write_bytes(payload)
    return table.num_rows


def _download_one(archive: Archive, *, dry_run: bool, keep_archive: bool) -> tuple[str, int, str]:
    if archive.output.exists() and not dry_run:
        return archive.period, 0, "cached"
    if dry_run:
        return archive.period, 0, archive.url
    payload = _request(archive.url)
    if payload is None:
        return archive.period, 0, "missing"
    _checksum(archive.url, payload)
    return archive.period, _write_part(archive, payload, keep_archive=keep_archive), "downloaded"


def _archives(root: Path, symbols: Iterable[str], intervals: Iterable[str], start: date, end: date) -> list[Archive]:
    result = []
    for symbol in symbols:
        for interval in intervals:
            key = _bar_key(symbol, interval)
            output_dir = root / "data" / "bars" / key
            for month in _iter_months(start, end):
                period = month.strftime("%Y-%m")
                result.append(Archive(symbol, interval, period, _archive_url(symbol, interval, period, "monthly"), output_dir / f"{period}.parquet"))
    return result


def _download_month(archive: Archive, *, dry_run: bool, keep_archive: bool) -> tuple[str, int, str]:
    result = _download_one(archive, dry_run=dry_run, keep_archive=keep_archive)
    if result[2] != "missing" or dry_run:
        return result
    # Monthly archives can be absent for a newly listed contract or a partial
    # month. Recover the same month from daily files without losing the key.
    month = _month_cursor(archive.period)
    last = date(month.year, month.month, calendar.monthrange(month.year, month.month)[1])
    today = datetime.now(UTC).date()
    if month.year == today.year and month.month == today.month:
        # Vision cannot publish future daily files. The current day's archive
        # is normally incomplete as well, so stop at the previous UTC day.
        last = min(last, today - timedelta(days=1))
    if last < month:
        return archive.period, 0, "pending"
    rows = 0
    downloaded = 0
    for day in (month + timedelta(days=i) for i in range((last - month).days + 1)):
        period = day.strftime("%Y-%m-%d")
        daily = Archive(archive.symbol, archive.interval, period, _archive_url(archive.symbol, archive.interval, period, "daily"), archive.output.with_name(f"{period}.parquet"))
        _, count, status = _download_one(daily, dry_run=False, keep_archive=keep_archive)
        if status == "downloaded":
            rows += count
            downloaded += 1
    return archive.period, rows, f"daily:{downloaded}"


def download(*, root: Path, symbols: Iterable[str] = DEFAULT_SYMBOLS, intervals: Iterable[str] = DEFAULT_INTERVALS, start: str = "2019-01", end: str | None = None, workers: int = 4, dry_run: bool = False, keep_archives: bool = False) -> None:
    """Download Vision files into the catalog; this function is reusable in tests."""
    root = Path(root).expanduser()
    symbols = tuple(_symbol(symbol) for symbol in symbols)
    intervals = tuple(intervals)
    unknown = sorted(set(intervals) - set(INTERVALS))
    if unknown:
        raise ValueError(f"unsupported interval(s): {', '.join(unknown)}")
    begin = _month_cursor(start)
    finish = _month_cursor(end) if end else datetime.now(UTC).date().replace(day=1)
    if begin >= finish:
        raise ValueError("start must be before end")
    archives = _archives(root, symbols, intervals, begin, finish)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "data" / "bars" / "_binance_vision.json"
    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(_download_month, item, dry_run=dry_run, keep_archive=keep_archives) for item in archives]
        for future in as_completed(futures):
            results.append(future.result())
            period, rows, status = results[-1]
            print(f"{period}: {status} ({rows:,} rows)")
    if not dry_run:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({"source": BASE_URL, "symbols": symbols, "intervals": intervals, "start": begin.isoformat(), "end": finish.isoformat(), "generated_at": datetime.now(UTC).isoformat()}, indent=2) + "\n")
    print(f"planned={len(archives)} completed={sum(status in ('downloaded', 'cached') or status.startswith('daily:') for _, _, status in results)} root={root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("~/.cache/alphax/catalog"))
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--intervals", nargs="+", default=DEFAULT_INTERVALS, choices=tuple(INTERVALS))
    parser.add_argument("--start", default="2019-01", help="first month, YYYY-MM")
    parser.add_argument("--end", help="exclusive month, YYYY-MM; defaults to current month")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-archives", action="store_true", help="retain downloaded ZIP files beside each Parquet part")
    args = parser.parse_args()
    download(root=args.root, symbols=args.symbols, intervals=args.intervals, start=args.start, end=args.end, workers=args.workers, dry_run=args.dry_run, keep_archives=args.keep_archives)


if __name__ == "__main__":
    main()
