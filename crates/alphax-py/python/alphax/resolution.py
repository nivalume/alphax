"""Simulating on finer bars than the signals were computed on.

A stop tested on hourly bars cannot tell whether the bar reached its
stop-loss or its take-profit first; on minute bars it can. The engine runs on
whatever bars it is given and has no timestamps, so the two resolutions meet
here:

- :func:`align` puts values computed on coarse bars onto fine bars, each at
  the first fine bar at or after the moment the coarse value became
  available — its bar's close, or ``ts_init``. Pass the times the values
  became known, not the times their bars opened: an hourly bar labelled by
  its open would otherwise leak an hour.
- :func:`resample` aggregates a fine series — a value curve, returns, cash
  flows — back onto coarse bars, to read a fine simulation at the
  resolution it was decided on.

::

    entries_1m = alphax.resolution.align(entries_1h, hour_close_times, minute_times)
    pf = alphax.from_signals(close_1m, entries_1m, exits_1m, sl_stop=0.01, tp_stop=0.02,
                             open=open_1m, high=high_1m, low=low_1m)
    value_1h = alphax.resolution.resample(pf.value(), minute_times, hour_close_times)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rates import _timestamps

__all__ = ["align", "resample"]


def _sorted_times(index: Any, name: str) -> np.ndarray:
    times = _timestamps(index)
    if (np.diff(times) <= 0).any():
        raise ValueError(f"{name} must be strictly increasing")
    return times


def _rows(values: Any, n: int, name: str) -> tuple[np.ndarray, bool]:
    array = np.asarray(values)
    if array.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1-D or 2-D; got {array.ndim} dimensions")
    if array.shape[0] != n:
        raise ValueError(f"{name} has {array.shape[0]} rows for {n} times")
    one_d = array.ndim == 1
    return (array.reshape(-1, 1) if one_d else array), one_d


def align(values: Any, available_time: Any, target_time: Any, *, fill: bool = False) -> np.ndarray:
    """Coarse values on fine bars.

    Each row of ``values`` lands on the first row of ``target_time`` at or
    after its ``available_time``; rows available after the last target are
    dropped. Booleans are events: ``True`` on the bar it lands on, ``False``
    elsewhere, two that land on one bar OR-ed. Numbers land as they are, the
    later of two on one bar winning, and ``NaN`` elsewhere — or, with
    ``fill``, the last value that landed, as a level that holds until the
    next (a stop distance, a target size).

    Parameters
    ----------
    values
        ``(n_coarse,)`` or ``(n_coarse, n_cols)``.
    available_time, target_time
        Strictly increasing ``datetime64`` arrays or pandas ``DatetimeIndex``
        es: when each coarse value became known, and the fine bars' times.

    Returns
    -------
    numpy.ndarray
        ``(n_fine,)`` or ``(n_fine, n_cols)``.
    """
    source = _sorted_times(available_time, "available_time")
    target = _sorted_times(target_time, "target_time")
    array, one_d = _rows(values, len(source), "values")
    at = np.searchsorted(target, source, side="left")
    lands = at < len(target)
    if array.dtype == np.dtype(bool):
        out = np.zeros((len(target), array.shape[1]), dtype=bool)
        np.logical_or.at(out, at[lands], array[lands])
    else:
        out = np.full((len(target), array.shape[1]), np.nan)
        out[at[lands]] = array[lands].astype(np.float64)
        if fill:
            landed = np.zeros(len(target), dtype=bool)
            landed[at[lands]] = True
            last = np.where(landed, np.arange(len(target)), -1)
            np.maximum.accumulate(last, out=last)
            held = last >= 0
            out[held] = out[last[held]]
    return out[:, 0] if one_d else out


def resample(values: Any, source_time: Any, target_time: Any, *, how: str = "last") -> np.ndarray:
    """A fine series on coarse bars.

    Coarse bar ``k`` takes the fine rows in ``(target_time[k - 1],
    target_time[k]]`` (the first, everything up to ``target_time[0]``):

    - ``"last"``: the last fine value at or before ``target_time[k]`` — a
      level such as the value curve, as of the coarse bar. ``NaN`` before the
      first fine row.
    - ``"sum"``: the sum over the bucket — cash flows, fees. 0 for an empty
      bucket.
    - ``"compound"``: ``prod(1 + r) − 1`` over the bucket — returns. 0 for an
      empty bucket.

    Returns ``(n_coarse,)`` or ``(n_coarse, n_cols)``.
    """
    source = _sorted_times(source_time, "source_time")
    target = _sorted_times(target_time, "target_time")
    array, one_d = _rows(values, len(source), "values")
    array = array.astype(np.float64)
    upto = np.searchsorted(source, target, side="right")
    before = np.concatenate(([0], upto[:-1]))
    if how == "last":
        out = np.full((len(target), array.shape[1]), np.nan)
        some = upto > 0
        out[some] = array[upto[some] - 1]
    elif how == "sum":
        running = np.vstack([np.zeros((1, array.shape[1])), np.cumsum(array, axis=0)])
        out = running[upto] - running[before]
    elif how == "compound":
        with np.errstate(divide="ignore"):
            logs = np.log1p(array)
        running = np.vstack([np.zeros((1, array.shape[1])), np.cumsum(logs, axis=0)])
        out = np.expm1(running[upto] - running[before])
    else:
        raise ValueError(f"how must be 'last', 'sum' or 'compound'; got {how!r}")
    return out[:, 0] if one_d else out
