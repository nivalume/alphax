"""Run a moving-average sweep over cached Binance Spot klines."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import alphax
import binance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


@alphax.strategy
def ma_cross(data, fast=10, slow=50):
    values = alphax.factors(
        data,
        fast=f"Mean($close, {fast})",
        slow=f"Mean($close, {slow})",
    )
    return (
        alphax.crossed_above(values["fast"], values["slow"]),
        alphax.crossed_above(values["slow"], values["fast"]),
    )


def load_panel(instruments, interval, directory):
    fields = [(name, pa.float64()) for name in ("open", "high", "low", "close", "volume")]
    schema = alphax.data.DataSchema(
        name="bars",
        input=pa.schema([("close_time", pa.uint64()), *fields]),
        output=pa.schema([("time", pa.uint64()), *fields]),
        event_time="time", available_time="time", decode=decode_cached,
    )
    unit = {"m": "MINUTE", "h": "HOUR", "d": "DAY", "w": "WEEK"}[interval[-1]]
    return alphax.data.Source({
        f"{i}-{int(interval[:-1])}-{unit}-LAST-EXTERNAL": binance._cache_path(directory, str(i), interval)
        for i in instruments
    }, schema).prepare()


def decode_cached(batch, context):
    # Binance's inclusive closing millisecond becomes the end boundary.
    time = pc.call_function(
        "multiply",
        [
            pc.call_function("add", [batch["close_time"], pa.scalar(1, pa.uint64())]),
            pa.scalar(1_000_000, pa.uint64()),
        ],
    )
    fields = ["open", "high", "low", "close", "volume"]
    return pa.RecordBatch.from_arrays([time, *[batch[n] for n in fields]], names=["time", *fields])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="BTCUSDT.BINANCE,ETHUSDT.BINANCE,SOLUSDT.BINANCE")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--dir", default="data/binance")
    parser.add_argument("--demo", action="store_true", help="Use temporary offline sample caches")
    args = parser.parse_args()

    symbols = tuple(
        symbol.strip() for symbol in args.symbols.split(",") if symbol.strip()
    )
    instruments = tuple(alphax.InstrumentId.parse(s) for s in symbols)
    temporary = TemporaryDirectory(prefix="alphax-crypto-") if args.demo else None
    try:
        directory = Path(temporary.name if temporary else args.dir)
        if args.demo:
            duration = binance._interval_ms(args.interval)
            for col, instrument in enumerate(instruments):
                rows = []
                for i in range(400):
                    close = 100 + col*30 + 10*np.sin(i/10)
                    rows.append(binance.Kline(i*duration, close, close+1, close-1, close, 50., (i+1)*duration-1))
                binance._save_cached(binance._cache_path(directory, str(instrument), args.interval), rows)
        panel = load_panel(instruments, args.interval, directory)
    finally:
        if temporary:
            temporary.cleanup()

    pairs = ((10, 50), (20, 50), (20, 100), (50, 200))
    portfolio = ma_cross.sweep(
        panel,
        params=[{"fast": fast, "slow": slow} for fast, slow in pairs],
        init_cash=10_000.0,
        fees=0.001,
        slippage=0.0005,
        ann_factor=binance.bars_per_year(args.interval),
        threads=4,
    )

    labels = [(fast, slow, symbol) for fast, slow in pairs for symbol in symbols]
    sharpe = portfolio.traded_only(portfolio.sharpe_ratio())
    total = portfolio.total_return()
    drawdown = portfolio.max_drawdown()
    trades = portfolio.trade_count()
    print(f"{panel.n_rows} bars x {panel.n_cols} symbols, coverage={panel.coverage()}")
    for index, (fast, slow, symbol) in enumerate(labels):
        print(
            f"fast={fast:<3} slow={slow:<3} {symbol:<10} trades={trades[index]:<4} "
            f"return={total[index]:>8.1%} maxDD={drawdown[index]:>8.1%} "
            f"sharpe={sharpe[index]:>7.2f}"
        )


if __name__ == "__main__":
    main()
