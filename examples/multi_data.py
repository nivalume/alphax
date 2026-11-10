"""Offline Bars + funding + news, two instruments and four parameter sets.

python examples/multi_data.py --source catalog  # also parquet / csv
Without --dir, samples live only in a TemporaryDirectory. With --dir, read
existing inputs only: data/{schema}/{key}/*.parquet for catalog, or
{schema}-{column}.{parquet,csv} for standalone files (columns 0 and 1).
"""
import argparse
import resource
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import alphax
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import csv, parquet

INSTRUMENTS = ("ADAUSDC.BINANCE", "ETHUSDC.BINANCE")
BAR_TYPES = tuple(f"{i}-1-MINUTE-LAST-EXTERNAL" for i in INSTRUMENTS)
PAIRS = ((10, 50), (20, 50), (20, 100), (50, 200))
OHLCV = ("open", "high", "low", "close", "volume")


def decode(batch, context):
    """User-owned conversion: vendor milliseconds -> availability nanoseconds."""
    names = [name for name in batch.schema.names if name not in ("timestamp_ms", "published_ms")]
    return pa.RecordBatch.from_arrays(
        [pc.call_function("multiply", [batch["timestamp_ms"], pa.scalar(1_000_000, pa.uint64())]),
         pc.call_function("multiply", [batch["published_ms"], pa.scalar(1_000_000, pa.uint64())]),
         *[batch[name] for name in names]],
        names=["event_time", "available_time", *names],
    )


def schemas():
    fields = {
        "bars": [(name, pa.float64()) for name in OHLCV],
        "funding": [("rate", pa.float64())],
        "news": [("headline", pa.string()), ("sentiment", pa.float64())],
    }
    return [alphax.data.DataSchema(
        name=name,
        input=pa.schema([("timestamp_ms", pa.uint64()), ("published_ms", pa.uint64()), *extra]),
        output=pa.schema([("event_time", pa.uint64()), ("available_time", pa.uint64()), *extra]),
        decode=decode,
    ) for name, extra in fields.items()]


def write_samples(root, source):
    """Example-only writer; the library's Source and Catalog never write."""
    rng = np.random.default_rng(20260920)
    times = np.arange(1000, dtype=np.uint64) * 60_000 + 1_700_000_000_000
    for col, instrument in enumerate(INSTRUMENTS):
        close = (1 if col == 0 else 2000) * np.exp(np.cumsum(rng.normal(0.0002, 0.005, times.size)))
        open_ = np.r_[close[0], close[:-1]]
        for name in ("bars", "funding", "news"):
            step = {"bars": 1, "funding": 80, "news": 13}[name]
            t = times[::step]
            # News published two minutes after the event cannot enter earlier factors.
            content: dict[str, Any] = {"timestamp_ms": t, "published_ms": t + (120_000 if name == "news" else 0)}
            if name == "bars":
                content.update(open=open_, high=np.maximum(open_, close)*1.01,
                               low=np.minimum(open_, close)*0.99, close=close,
                               volume=rng.uniform(10, 100, t.size), ignored_vendor_field=np.ones(t.size))
            elif name == "funding":
                content["rate"] = np.sin(np.arange(t.size)) * 0.00015
            else:
                content.update(headline=[f"Synthetic event {i} for {instrument}" for i in range(t.size)],
                               sentiment=np.cos(np.arange(t.size)/3))
            table = pa.table(content)
            key = BAR_TYPES[col] if name == "bars" else instrument
            path = (root / "data" / name / key / "part.parquet" if source == "catalog"
                    else root / f"{name}-{col}.{source}")
            path.parent.mkdir(parents=True, exist_ok=True)
            if source == "csv":
                csv.write_csv(table, path)
            else:
                parquet.write_table(table, path)


def crossed_above(left, right):
    valid = np.isfinite(left) & np.isfinite(right)
    above = left > right
    crossed = np.zeros_like(above)
    crossed[1:] = valid[1:] & valid[:-1] & above[1:] & ~above[:-1]
    return crossed


def delayed(signals):
    result = np.zeros_like(signals)
    result[1:] = signals[:-1]
    return result


def peak_rss_mib():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024**2 if sys.platform == "darwin" else 1024)


def run(root, source):
    timings = {}
    started = perf_counter()
    definitions = schemas()
    loaded = {}
    for definition in definitions:
        keys = BAR_TYPES if definition.name == "bars" else INSTRUMENTS
        # The catalog layout is the caller's: a schema directory holds one
        # directory per instrument, and Source joins the two.
        data = (alphax.data.Source(root / "data" / definition.name, definition, keys) if source == "catalog" else
                alphax.data.Source({key: root / f"{definition.name}-{col}.{source}" for col, key in enumerate(keys)}, definition))
        loaded[definition.name] = data.load()
    timings["load_decode"] = perf_counter() - started
    timings["load_decode_peak_rss_mib"] = peak_rss_mib()
    started = perf_counter()
    panel = loaded["bars"].prepare()
    timings["bar_panel"] = perf_counter() - started
    timings["bar_panel_peak_rss_mib"] = peak_rss_mib()
    started = perf_counter()
    fields = {
        "close": panel.close,
        "funding_rate": loaded["funding"].asof(panel, field="rate"),
        "news_sentiment": loaded["news"].window(panel, field="sentiment", window="30m", agg="mean", empty=0.0),
    }
    timings["align_aux"] = perf_counter() - started
    timings["align_aux_peak_rss_mib"] = peak_rss_mib()
    expressions = {"allow_entry": "($funding_rate < 0.0001) & ($news_sentiment >= 0)"}
    for index, (fast, slow) in enumerate(PAIRS):
        expressions[f"fast_{index}"] = f"Mean($close, {fast})"
        expressions[f"slow_{index}"] = f"Mean($close, {slow})"
    started = perf_counter()
    values = alphax.FactorSet.compile(expressions).evaluate(fields)
    timings["factors"] = perf_counter() - started
    timings["factors_peak_rss_mib"] = peak_rss_mib()
    entries = delayed(np.concatenate([
        crossed_above(values[f"fast_{i}"], values[f"slow_{i}"]) & (values["allow_entry"] > 0)
        for i in range(len(PAIRS))], axis=1))
    exits = delayed(np.concatenate([
        crossed_above(values[f"slow_{i}"], values[f"fast_{i}"])
        for i in range(len(PAIRS))], axis=1))
    column_map = np.tile(np.arange(panel.n_cols), len(PAIRS))
    started = perf_counter()
    portfolio = alphax.from_signals(
        panel.close, entries, exits, column_map=column_map,
        init_cash=10_000.0, fees=0.001, slippage=0.0005,
        ann_factor=365*24*60, threads=4,
    )
    timings["simulation"] = perf_counter() - started
    timings["simulation_peak_rss_mib"] = peak_rss_mib()
    assert panel.n_cols == 2
    assert entries.shape == (panel.n_rows, 8) and portfolio.n_cols == 8
    assert portfolio.price_nbytes == panel.close.nbytes
    return panel, fields, values, entries, exits, portfolio, timings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("catalog", "parquet", "csv"), default="catalog")
    parser.add_argument("--dir", type=Path, help="Read existing inputs; never write samples here")
    args = parser.parse_args()
    if args.dir is None:
        with TemporaryDirectory(prefix="alphax-demo-") as directory:
            root = Path(directory)
            write_samples(root, args.source)
            result = run(root, args.source)
    else:
        result = run(args.dir, args.source)
    panel, _, _, entries, _, portfolio, timings = result
    print(f"{args.source}: close={panel.shape}, entries/exits={entries.shape}")
    print("column_map:", np.tile(np.arange(panel.n_cols), len(PAIRS)).tolist())
    print("labels:", [(fast, slow, str(i)) for fast, slow in PAIRS for i in panel.instruments])
    print("returns:", portfolio.total_return())
    print("stage_ms:", {k: round(v*1000, 3) for k,v in timings.items() if not k.endswith("_peak_rss_mib")})
    print(f"source_price_bytes={portfolio.price_nbytes}; expanded_price_bytes={panel.close.nbytes*len(PAIRS)}")
    print("process_peak_rss_mib_after_stage:", {k.removesuffix("_peak_rss_mib"): round(v, 2) for k,v in timings.items() if k.endswith("_peak_rss_mib")})


if __name__ == "__main__":
    main()
