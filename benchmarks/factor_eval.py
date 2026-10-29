#!/usr/bin/env python3
"""Wall-clock timing of factor evaluation.

An Alpha158-scale panel: 2520 rows by 100 symbols (pass 500 as the second
argument), 10 factors, horizons 1, 5, 10 and 20. The clock starts after the
panels exist, on arrays that are already column-major, so the number is the
evaluation and not the copy into it.

Elements, in the throughput column, are the values the call scores:

- ``forward_returns``: bars x symbols x horizons
- ``cs_ic`` and ``evaluate``: bars x symbols x factors x horizons
- quantile labels and returns: bars x symbols

A ratio near 4 between one thread and four is the scaling the kernels are
built for. Absolute milliseconds move with the machine; the ratio is the
thing to read. A checksum of the rank-IC cube is printed so a run that
returned a different panel is obvious.

    uv run python benchmarks/factor_eval.py
    uv run python benchmarks/factor_eval.py 2520 500 10
"""

from __future__ import annotations

import sys
import time

import numpy as np
import phasor


ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2520
SYMBOLS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
N_FACTORS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
HORIZONS = (1, 5, 10, 20)
RUNS = 3


def best_of(runs: int, fn):
    best = float("inf")
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def million_per_sec(count: int, seconds: float) -> float:
    return count / seconds / 1_000_000.0


def report(name: str, seconds: float, count: int) -> None:
    print(f"{name:<42} {seconds * 1e3:9.1f} ms   {million_per_sec(count, seconds):8.2f} M elem/s")


def main() -> None:
    rng = np.random.default_rng(20261029)
    close = np.asfortranarray(100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, (ROWS, SYMBOLS)), axis=0))
    shock = rng.normal(size=(ROWS, SYMBOLS))
    factors = [
        np.asfortranarray(shock * (0.2 * k) + rng.normal(scale=1.0, size=(ROWS, SYMBOLS)))
        for k in range(N_FACTORS)
    ]
    named = {f"f{k}": factors[k] for k in range(N_FACTORS)}
    panel = ROWS * SYMBOLS
    ic_elems = panel * N_FACTORS * len(HORIZONS)
    forward_elems = panel * len(HORIZONS)
    print(
        f"panel {ROWS} x {SYMBOLS}, {N_FACTORS} factors, horizons {HORIZONS}\n"
        f"best of {RUNS}; elements are the values scored, not the bytes copied\n"
    )

    # One untimed pass so the first timed call is not the one that faults the pages in.
    phasor.factor_eval.forward_returns(close, horizons=HORIZONS, lag=1)
    phasor.factor_eval.cs_ic(factors, [factors[0]], min_names=8, threads=1)

    secs, forwards = best_of(RUNS, lambda: phasor.factor_eval.forward_returns(close, horizons=HORIZONS, lag=1))
    report("forward_returns (4 horizons)", secs, forward_elems)

    fwd_panels = [forwards[h] for h in HORIZONS]
    secs, _ = best_of(RUNS, lambda: phasor.factor_eval.cs_ic(factors, fwd_panels, min_names=8, threads=1))
    report("cs_ic (1 thread)", secs, ic_elems)
    one = secs
    secs, ic = best_of(RUNS, lambda: phasor.factor_eval.cs_ic(factors, fwd_panels, min_names=8, threads=4))
    report("cs_ic (4 threads)", secs, ic_elems)
    print(f"{'cs_ic scaling 1 -> 4':<42} {one / secs:9.2f} x")

    secs, labels = best_of(RUNS, lambda: phasor.factor_eval.quantile_labels(factors[0], q=5))
    report("quantile_labels (q=5)", secs, panel)
    secs, groups = best_of(
        RUNS, lambda: phasor.factor_eval.quantile_returns(labels, forwards[1], q=5)
    )
    report("quantile_returns (h=1, q=5)", secs, panel)

    secs, _ = best_of(
        RUNS,
        lambda: phasor.factor_eval.evaluate(named, close, horizons=HORIZONS, min_names=8, threads=1),
    )
    report("evaluate (1 thread)", secs, ic_elems)
    one = secs
    secs, evaluated = best_of(
        RUNS,
        lambda: phasor.factor_eval.evaluate(named, close, horizons=HORIZONS, min_names=8, threads=4),
    )
    report("evaluate (4 threads)", secs, ic_elems)
    print(f"{'evaluate scaling 1 -> 4':<42} {one / secs:9.2f} x")
    print(f"\nchecksum nanmean(rank IC) {float(np.nanmean(evaluated.spearman)):.6f}")
    print(f"checksum nanmean(cs spearman) {float(np.nanmean(ic[1])):.6f}")
    print(f"checksum nanmean(quantile return) {float(np.nanmean(groups)):.6f}")


if __name__ == "__main__":
    main()
