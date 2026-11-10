"""Synthetic prices with known properties, to build controls with.

A backtest on real prices cannot tell a strategy that found an edge from one
that fit the noise, because the true process is unknown. A :func:`gbm` with
``mu=0`` has no edge by construction, so a strategy that reports a healthy
Sharpe on it is measuring its own overfitting. These exist to build that
control, not to model a market.

Every draw is a function of ``(seed, row, col)``: the same arguments give the
same path bit for bit, and column ``c`` does not depend on how many columns
were asked for. ``shape`` is ``n`` for one 1-D series or ``(n_rows, n_cols)``
for independent paths side by side.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _alphax
from ._columns import columns, restore, shape as _shape

__all__ = ["gbm", "ohlc_from_close", "random_walk"]


def random_walk(shape: Any, *, start: float = 100.0, step_std: float = 1.0, seed: int = 0) -> np.ndarray:
    """Gaussian random walks: each bar adds ``step_std`` times a standard normal.

    The first bar is ``start``. Steps are additive, so a path can go negative;
    :func:`gbm` stays positive.
    """
    n_rows, n_cols, one_d = _shape(shape)
    out = _alphax.random_walk(n_rows, n_cols, float(start), float(step_std), int(seed))
    return restore(out, one_d)


def gbm(
    shape: Any,
    *,
    start: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.2,
    dt: float = 1 / 252,
    seed: int = 0,
) -> np.ndarray:
    """Geometric Brownian motion, discretized exactly.

    ``S[t+1] = S[t] * exp((mu - sigma**2 / 2) * dt + sigma * sqrt(dt) * Z)``:
    the closed-form step, not an Euler approximation, so log returns are
    exactly normal whatever ``dt`` is. ``mu`` and ``sigma`` are per unit of
    time and ``dt`` is one bar in that unit — by default annual figures over
    daily bars. The default ``mu=0`` is the no-edge control.
    """
    n_rows, n_cols, one_d = _shape(shape)
    out = _alphax.gbm(n_rows, n_cols, float(start), float(mu), float(sigma), float(dt), int(seed))
    return restore(out, one_d)


def ohlc_from_close(close: Any, *, spread: float = 0.01, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Opens, highs and lows consistent with ``close``.

    Returns ``(open, high, low)`` shaped like ``close``. The open is the
    previous close (the first bar opens at its own); the high and low widen
    ``max(open, close)`` and ``min(open, close)`` by up to ``spread`` of their
    magnitude, drawn uniformly. ``low <= open, close <= high`` holds on every
    bar without NaN. ``spread`` must lie in ``[0, 1)``.
    """
    arr, one_d = columns(close, "close")
    open_, high, low = _alphax.ohlc_from_close(arr, float(spread), int(seed))
    return restore(open_, one_d), restore(high, one_d), restore(low, one_d)
