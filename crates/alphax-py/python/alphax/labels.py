"""Supervised-learning targets: forward returns, trend, triple-barrier and
zig-zag pivot labels.

**Every function here reads the future.** These are the ``y`` a model is
trained to predict: the value at bar ``t`` is computed from bars after ``t``.
None of them is a signal, and none may be fed to a simulation as entries,
exits or features — a backtest that does so is not optimistic, it is wrong.
Each function says how far ahead it looks, which is the gap a train/test
split must leave between its halves.

``close`` is ``(bars,)`` or ``(bars, columns)`` and the result has the same
shape. Columns never see each other: a label near the end of one column does
not borrow bars from the next.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _alphax
from ._columns import columns, restore

__all__ = ["fixed_horizon_returns", "mean_labels", "pivot_labels", "trend_labels", "triple_barrier"]


def fixed_horizon_returns(close: Any, horizon: int) -> np.ndarray:
    """``close[t + horizon] / close[t] - 1``; NaN where that runs past the end.

    Looks ``horizon`` bars ahead.
    """
    arr, one_d = columns(close, "close")
    return restore(_alphax.fixed_horizon_returns(arr, int(horizon)), one_d)


def trend_labels(close: Any, horizon: int, threshold: float = 0.0) -> np.ndarray:
    """+1, -1 or 0 by whether the ``horizon``-bar forward return is above
    ``threshold``, below ``-threshold``, or neither.

    ``int8``. Where the return is undefined the label is 0, since a class has
    no NaN; mask with :func:`fixed_horizon_returns` to drop those rows. Looks
    ``horizon`` bars ahead.
    """
    arr, one_d = columns(close, "close")
    return restore(_alphax.trend_labels(arr, int(horizon), float(threshold)), one_d)


def triple_barrier(close: Any, *, up: float, down: float, max_hold: int) -> tuple[np.ndarray, np.ndarray]:
    """Which barrier each bar's close meets first, and when.

    From each bar ``t`` the later closes are walked for up to ``max_hold``
    bars: a return of at least ``up`` labels +1, one of at most ``-down``
    labels -1, and reaching the time limit (or the column's end) labels 0.

    Returns ``(labels, decided_at)``: ``int8`` labels, and the row within the
    column where each was decided — -1 on the column's last bar or a NaN
    close. Barriers are on closes, so a bar whose high touched ``up`` but
    closed below it does not count. Looks up to ``max_hold`` bars ahead, and
    ``decided_at`` says exactly how far each label did.
    """
    arr, one_d = columns(close, "close")
    labels, decided = _alphax.triple_barrier(arr, float(up), float(down), int(max_hold))
    return restore(labels, one_d), restore(decided, one_d)


def pivot_labels(close: Any, threshold: float) -> np.ndarray:
    """+1 at confirmed zig-zag troughs, -1 at confirmed peaks, 0 elsewhere.

    A trough is confirmed once a later close is at least ``threshold`` above
    it, a peak once one is at least ``threshold`` below; the last, unconfirmed
    extreme stays 0. Non-zero labels alternate. ``int8``.

    Looks ahead without bound: a pivot is known only once something
    arbitrarily later confirms it. "Buy at the zig-zag trough" backtests
    beautifully and cannot be traded.
    """
    arr, one_d = columns(close, "close")
    return restore(_alphax.pivot_labels(arr, float(threshold)), one_d)


def mean_labels(close: Any, window: int, *, wait: int = 1) -> np.ndarray:
    """The mean of the ``window`` closes starting ``wait`` bars ahead, over
    this close, less 1: ``mean(close[t + wait : t + wait + window]) / close[t]
    - 1``. Smoother than one forward return, so a label less driven by the
    one bar the horizon ends on. NaN where the window runs past the end, or
    has a NaN in it. Looks ``wait + window - 1`` bars ahead."""
    if window < 1 or wait < 0:
        raise ValueError("window must be 1 or more and wait 0 or more")
    c, one_d = columns(close, "close")
    n = c.shape[0]
    out = np.full_like(c, np.nan)
    csum = np.vstack([np.zeros((1, c.shape[1])), np.cumsum(c, axis=0)])
    last = n - wait - window
    if last >= 0:
        t = np.arange(last + 1)
        mean = (csum[t + wait + window] - csum[t + wait]) / window
        out[t] = mean / c[t] - 1
    return restore(np.asfortranarray(out), one_d)
