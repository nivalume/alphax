#!/usr/bin/env python3
"""Wall-clock timing of the Python API.

The Rust benchmarks time the loops; nothing timed what a Python caller pays
around them until this. Each line is the best of five on synthetic data::

    uv run python benchmarks/bench_python.py            # default 2000 x 50, 8 params
    uv run python benchmarks/bench_python.py 5000 200 8

The rows are deliberately the calls a research loop makes: build signals with
factors and crossed_above, run a sweep, then read several metrics and a
per-column stats loop off the same Portfolio. Ratios between rows matter more
than absolute numbers; the machine is shared.
"""

from __future__ import annotations

import sys
import time

import alphax
import numpy as np

N_ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
N_SYMBOLS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
N_PARAMS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
THREADS = 4


def best_of(runs, fn):
    best = float("inf")
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def report(name, secs):
    print(f"{name:<44} {secs * 1e3:>9.1f} ms")


@alphax.strategy
def ma_cross(data, fast=5, slow=20):
    values = alphax.factors(data, fast=f"Mean($close, {fast})", slow=f"Mean($close, {slow})")
    return (
        alphax.crossed_above(values["fast"], values["slow"]),
        alphax.crossed_above(values["slow"], values["fast"]),
    )


def main() -> None:
    rng = np.random.default_rng(20260927)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.011, (N_ROWS, N_SYMBOLS)), axis=0)
    close_f = np.asfortranarray(close)
    data = {"close": close_f}
    params = [{"fast": 5 + 5 * k, "slow": 20 + 20 * k} for k in range(N_PARAMS)]
    n_cols = N_SYMBOLS * N_PARAMS
    print(f"panel {N_ROWS} x {N_SYMBOLS}, {N_PARAMS} parameter sets -> {n_cols} columns, {THREADS} threads\n")

    secs, factors = best_of(5, lambda: alphax.factors(data, fast="Mean($close, 5)", slow="Mean($close, 20)"))
    report("factors (2 x Mean)", secs)
    secs, crossed = best_of(5, lambda: alphax.crossed_above(factors["fast"], factors["slow"]))
    report("crossed_above", secs)

    secs, pf = best_of(5, lambda: ma_cross.sweep(data, params=params, fees=0.001, threads=THREADS))
    report(f"strategy.sweep ({N_PARAMS} params, C-order close)", secs)

    entries = np.asfortranarray(np.tile(crossed, (1, N_PARAMS)))
    exits = np.asfortranarray(np.tile(alphax.crossed_above(factors["slow"], factors["fast"]), (1, N_PARAMS)))
    column_map = np.tile(np.arange(N_SYMBOLS), N_PARAMS)
    secs, _ = best_of(
        5, lambda: alphax.from_signals(close_f, entries, exits, column_map=column_map, fees=0.001, threads=THREADS)
    )
    report("from_signals (F-order, column_map)", secs)
    secs, _ = best_of(
        5, lambda: alphax.from_signals(close_f, entries, exits, column_map=column_map, fees=0.001, threads=1)
    )
    report("from_signals (1 thread)", secs)

    secs, _ = best_of(5, pf.sharpe_ratio)
    report("Portfolio.sharpe_ratio()", secs)
    secs, _ = best_of(5, pf.total_return)
    report("Portfolio.total_return()", secs)
    secs, _ = best_of(5, pf.trade_count)
    report("Portfolio.trade_count()", secs)
    secs, _ = best_of(5, pf.metrics)
    report("Portfolio.metrics()", secs)
    secs, _ = best_of(3, pf.stats)
    report("Portfolio.stats() (all columns)", secs)
    per_column = min(16, n_cols)
    secs, _ = best_of(3, lambda: [pf.stats(c) for c in range(per_column)])
    report(f"Portfolio.stats(col) x {per_column}", secs)
    secs, _ = best_of(5, pf.trades)
    report("Portfolio.trades()", secs)
    secs, _ = best_of(5, pf.orders)
    report("Portfolio.orders()", secs)


if __name__ == "__main__":
    main()
