"""Local data in one path: a file, one instrument's directory, or a schema directory.

python examples/local_data.py            # deterministic samples under a temporary directory
python examples/local_data.py --dir DIR  # read an existing catalog; DIR is never written

Every dataset below is a schema directory holding one directory per instrument:

    DIR/data/bars/<BarType>/*.csv.gz
    DIR/data/funding/<InstrumentId>/part.parquet
    DIR/data/sentiment/global.csv

A path is resolved per key, first match wins: the path is a file; `path/<key>` is
a file; `path/<key>` is a directory, read in filename order; or the path holds one
key's files. Only content decides CSV, Parquet and gzip, so a file name never
carries data. `decode` is the only place vendor units live, and `available_time`
is when a row became knowable, so neither a range filter nor an asof view reads
ahead of it.
"""
import argparse
import gzip
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import alphax
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
import pyarrow.parquet as parquet

SYMBOLS = ("ADAUSDC.BINANCE", "ETHUSDC.BINANCE")
BAR_TYPES = tuple(f"{symbol}-1-DAY-LAST-EXTERNAL" for symbol in SYMBOLS)
OHLCV = ("open", "high", "low", "close", "volume")
DAY_MS = 86_400_000


def bars_schema():
    """Vendor milliseconds in, UTC nanoseconds out; a bar is knowable at its close."""
    traded = [(name, pa.float64()) for name in OHLCV]

    def decode(batch, context):
        closed = pc.multiply(batch["close_time_ms"].cast(pa.uint64()), pa.scalar(1_000_000, pa.uint64()))
        opened = pc.multiply(batch["open_time_ms"].cast(pa.uint64()), pa.scalar(1_000_000, pa.uint64()))
        return pa.RecordBatch.from_arrays(
            [opened, closed, *[batch[name] for name in OHLCV]],
            names=["event_time", "available_time", *OHLCV],
        )

    return alphax.data.DataSchema(
        name="bars",
        input=pa.schema([("open_time_ms", pa.int64()), ("close_time_ms", pa.int64()), *traded]),
        output=pa.schema([("event_time", pa.uint64()), ("available_time", pa.uint64()), *traded]),
        decode=decode,
    )


def funding_schema():
    """A funding rate is knowable at the moment it is charged."""
    return alphax.data.DataSchema(
        name="funding",
        input=pa.schema([("ts_event", pa.int64()), ("funding_rate", pa.float64()), ("mark_price", pa.float64())]),
        output=pa.schema([("event_time", pa.uint64()), ("available_time", pa.uint64()), ("funding_rate", pa.float64())]),
        decode=lambda batch, context: pa.RecordBatch.from_arrays(
            [batch["ts_event"].cast(pa.uint64()), batch["ts_event"].cast(pa.uint64()), batch["funding_rate"]],
            names=["event_time", "available_time", "funding_rate"],
        ),
    )


def sentiment_schema():
    """Market-wide news binds no instrument: `None` broadcasts it onto every column.

    `headline` stays in `output` so it survives in `Dataset.batches`; `prepare`
    and the aligned views ignore every field they were not asked for.
    """
    return alphax.data.DataSchema(
        name="sentiment",
        input=pa.schema([("ts_event", pa.int64()), ("headline", pa.string()), ("score", pa.float64())]),
        output=pa.schema([("event_time", pa.uint64()), ("available_time", pa.uint64()),
                          ("headline", pa.string()), ("score", pa.float64())]),
        decode=lambda batch, context: pa.RecordBatch.from_arrays(
            [batch["ts_event"].cast(pa.uint64()), batch["ts_event"].cast(pa.uint64()),
             batch["headline"], batch["score"]],
            names=["event_time", "available_time", "headline", "score"],
        ),
        instrument=None,
    )


def write_samples(root):
    """Example-only writer: the library itself never creates or writes a file."""
    rng = np.random.default_rng(20261018)
    days = 120
    opened = np.arange(days, dtype=np.int64) * DAY_MS + int(np.datetime64("2026-01-01", "ms").astype(np.int64))

    for column, symbol in enumerate(SYMBOLS):
        close = (1.0 if column == 0 else 2000.0) * np.exp(np.cumsum(rng.normal(0.0005, 0.02, days)))
        open_ = np.r_[close[0], close[:-1]]
        bars = pa.table({
            "open_time_ms": pa.array(opened, pa.int64()),
            "close_time_ms": pa.array(opened + DAY_MS - 1, pa.int64()),
            "open": pa.array(open_), "high": pa.array(np.maximum(open_, close) * 1.01),
            "low": pa.array(np.minimum(open_, close) * 0.99), "close": pa.array(close),
            "volume": pa.array(rng.uniform(1e3, 1e4, days)),
        })
        # Two files in one key's directory: a scan reads them in filename order.
        for part in range(2):
            path = root / "data" / "bars" / BAR_TYPES[column] / f"2026-part-{part}.csv.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wb") as handle:
                csv.write_csv(bars.slice(part * 60, 60), handle)

        funding = pa.table({
            "ts_event": pa.array(opened + DAY_MS // 3, pa.int64()),
            "funding_rate": pa.array(np.sin(np.arange(days) / 7) * 0.0002),
            # A nullable column is simply typed as nullable; empty cells stay null.
            "mark_price": pa.array(np.where(np.arange(days) % 5 == 0, None, close), pa.float64()),
        })
        path = root / "data" / "funding" / symbol / "part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        parquet.write_table(funding, path)

    news = pa.table({
        "ts_event": pa.array(opened[::10] + DAY_MS // 2, pa.int64()),
        "headline": pa.array([f"market note {i}" for i in range(len(opened[::10]))]),
        "score": pa.array(np.cos(np.arange(len(opened[::10])) / 3)),
    })
    path = root / "data" / "sentiment" / "global.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    csv.write_csv(news, path)


def stamp(nanoseconds):
    return np.datetime64(int(nanoseconds), "ns").astype("datetime64[s]")


def run(root):
    """Three granularities, one range, and two auxiliary series on the panel."""
    # 1. One path, three granularities, one panel. A key names the instrument:
    #    bars use a BarType, and the panel's columns follow the key order.
    keys = BAR_TYPES
    directory = alphax.data.load(root / "data" / "bars", bars_schema(), keys).prepare()
    one_key = alphax.data.load(root / "data" / "bars" / keys[0], bars_schema()).prepare()
    first_file = sorted((root / "data" / "bars" / keys[0]).iterdir())[0]
    first = alphax.data.load(first_file, bars_schema(), keys[0])
    one_file = first.prepare()
    print("panel", directory.shape, [str(one) for one in directory.instruments])
    print("one instrument", one_key.shape, "| its first file", one_file.shape)
    print("event -> available:", stamp(first.batches[0][1]["event_time"][0].as_py()),
          "->", stamp(first.batches[0][1]["available_time"][0].as_py()))
    np.testing.assert_array_equal(one_key.close[:, 0], directory.close[:, 0])
    np.testing.assert_array_equal(one_file.time, one_key.time[: one_file.n_rows])

    # 2. A range is half open on the decoded availability column, never on a file
    #    name. A Source keeps the resolved files for another read.
    source = alphax.data.Source(root / "data" / "bars", bars_schema(), keys)
    march = source.prepare(start="2026-03-01", end="2026-04-01")
    before = source.prepare(start=int(directory.time[0]), end=int(march.time[0]))
    print("2026-03:", march.n_rows, "rows | before it:", before.n_rows, "rows | coverage",
          directory.coverage().round(3))

    # 3. Auxiliary series: keyed by instrument, or one series for every column.
    funding = alphax.data.load(root / "data" / "funding", funding_schema(), SYMBOLS)
    news = alphax.data.load(root / "data" / "sentiment" / "global.csv", sentiment_schema(), "market")
    fields = {
        "close": directory.close,
        "funding_rate": funding.asof(directory, field="funding_rate"),
        "funding_3d": funding.window(directory, field="funding_rate", window="3d", agg="mean", empty=0.0),
        "sentiment": news.asof(directory, field="score"),
    }
    for name, values in fields.items():
        print(f"  {name:12} {values.shape} {values.dtype}")
    np.testing.assert_allclose(fields["sentiment"][:, 0], fields["sentiment"][:, 1])  # global broadcasts

    # 4. `batches` keeps everything `output` declared, including text.
    print("headlines:", news.batches[0][1]["headline"].to_pylist()[:2])
    return directory


def mean(close, window):
    """A trailing mean per column, NaN until the window is full."""
    sums = np.cumsum(np.vstack([np.zeros((1, close.shape[1])), close]), axis=0)
    out = np.full_like(close, np.nan)
    out[window - 1 :] = (sums[window:] - sums[:-window]) / window
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, help="An existing catalog in the layout above; never written")
    args = parser.parse_args()
    if args.dir is None:
        with TemporaryDirectory(prefix="alphax-data-") as temporary:
            root = Path(temporary)
            write_samples(root)
            panel = run(root)
    else:
        panel = run(args.dir)

    # 5. The panel is what the engine takes: market arrays stay [T, S].
    close = panel.close
    fast, slow = mean(close, 5), mean(close, 20)
    entries = np.zeros(close.shape, dtype=bool)
    exits = np.zeros(close.shape, dtype=bool)
    entries[1:] = (fast[1:] > slow[1:]) & (fast[:-1] <= slow[:-1])
    exits[1:] = (fast[1:] < slow[1:]) & (fast[:-1] >= slow[:-1])
    portfolio = alphax.from_signals(close, entries, exits, init_cash=10_000.0, fees=0.001, slippage=0.0005)
    print("engine:", portfolio.trade_count().tolist(), "trades, returns", portfolio.total_return().round(4).tolist())


if __name__ == "__main__":
    main()
