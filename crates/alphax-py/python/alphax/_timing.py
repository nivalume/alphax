"""Time-based stops and order expiries as the rows the engine counts.

The engine has no timestamps: ``td_stop`` is a number of bars after the
entry, ``dt_stop`` a row, ``limit_tif`` a number of bars after the order and
``limit_expiry`` the row it is cancelled at. With the bars' times these can
be given as time instead (``time_delta_format="index"``), and are converted
here, for every bar the position or order could start on:

- a duration (``"7D"``, ``"90min"``, a ``timedelta``) for ``td_stop`` and
  ``limit_tif``: a time stop exits at the open of the first bar at least
  that long after the entry bar's time; an order may fill on the bars
  before that long has passed since the bar it was placed on;
- for ``dt_stop`` and ``limit_expiry``, a moment (``"2024-03-01"``), a period
  (``"D"``, ``"W"``, ``"ME"``: the end of the period the bar is in) or a time
  of day (``"18:00"``: its next occurrence): the stop exits, and the order
  is cancelled, at the first bar at or after it.

A bar's time is the instant its timestamp marks, as given. Periods and times
of day are read in the index's time zone (UTC without one).
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

import numpy as np

from .rates import _timestamps

_CLOCK = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")


def is_time(value: Any) -> bool:
    """Whether a stop or expiry is given as time rather than rows."""
    if isinstance(value, (str, _dt.timedelta, _dt.datetime, _dt.date, np.timedelta64, np.datetime64)):
        return True
    array = np.asarray(value) if isinstance(value, (list, tuple, np.ndarray)) else None
    return array is not None and array.dtype.kind in "mMOU"


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - pandas is optional
        raise ImportError("time-based stops given as strings need pandas") from e
    return pd


def _duration_ns(value: Any) -> int:
    if isinstance(value, str):
        return int(_pandas().Timedelta(value).value)
    if isinstance(value, _dt.timedelta):
        return int(value / _dt.timedelta(microseconds=1)) * 1000
    if isinstance(value, np.timedelta64):
        return int(value.astype("timedelta64[ns]").astype(np.int64))
    raise TypeError(f"a duration must be a string, a timedelta or a timedelta64; got {value!r}")


def _per_column(value: Any, n_cols: int) -> list:
    if isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim == 1):
        values = list(value)
        if len(values) != n_cols:
            raise ValueError(f"one value per column: {len(values)} for {n_cols} columns")
        return values
    return [value] * n_cols


def _index(index: Any):
    """The index as pandas, and its times as UTC nanoseconds."""
    times = _timestamps(index)
    return index, times


def duration_rows(value: Any, index: Any, n_cols: int, *, inclusive: bool) -> np.ndarray:
    """``(n_rows, n_cols)`` rows from each bar to the first bar at least
    ``value`` later (``inclusive=False``: ``td_stop``), or to the last bar
    before that (``inclusive=True``: ``limit_tif``)."""
    _, times = _index(index)
    n = len(times)
    rows = np.arange(n)
    out = np.empty((n, n_cols))
    for col, v in enumerate(_per_column(value, n_cols)):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out[:, col] = np.nan
            continue
        reach = np.searchsorted(times, times + _duration_ns(v), side="left")
        out[:, col] = reach - rows - (1 if inclusive else 0)
    return np.asfortranarray(out)


def _deadlines(spec: Any, times: np.ndarray, tz: Any) -> np.ndarray:
    """For each bar, the moment ``spec`` names from it, as UTC nanoseconds."""
    if isinstance(spec, np.datetime64):
        return np.full(len(times), spec.astype("datetime64[ns]").astype(np.int64))
    pd = _pandas()
    if isinstance(spec, str) and _CLOCK.match(spec):
        clock = pd.Timedelta(spec if spec.count(":") == 2 else spec + ":00")
        local = pd.DatetimeIndex(times.astype("datetime64[ns]"), tz="UTC")
        if tz is not None:
            local = local.tz_convert(tz)
        at = local.normalize() + clock
        at = at.where(at >= local, at + pd.Timedelta(days=1))
        return np.asarray(at.tz_convert("UTC").tz_localize(None).asi8)
    if isinstance(spec, str):
        try:
            offset = pd.tseries.frequencies.to_offset(spec)
        except ValueError:
            offset = None
        if offset is not None:
            local = pd.DatetimeIndex(times.astype("datetime64[ns]"), tz="UTC")
            if tz is not None:
                local = local.tz_convert(tz)
            if isinstance(offset, pd.tseries.offsets.Tick):
                at = local.floor(offset) + offset
            else:
                at = pd.DatetimeIndex([offset.rollforward(t) for t in local.normalize()])
            return np.asarray(at.tz_convert("UTC").tz_localize(None).asi8)
    moment = pd.Timestamp(spec)
    if moment.tzinfo is None and tz is not None:
        moment = moment.tz_localize(tz)
    if moment.tzinfo is not None:
        moment = moment.tz_convert("UTC").tz_localize(None)
    return np.full(len(times), moment.value, dtype=np.int64)


def deadline_rows(value: Any, index: Any, n_cols: int) -> np.ndarray:
    """``(n_rows, n_cols)``: for each bar, the first row at or after the
    moment ``value`` names from it (``dt_stop``, ``limit_expiry``)."""
    _, times = _index(index)
    tz = getattr(index, "tz", None)
    out = np.empty((len(times), n_cols))
    for col, v in enumerate(_per_column(value, n_cols)):
        if v is None:
            out[:, col] = np.nan
            continue
        out[:, col] = np.searchsorted(times, _deadlines(v, times, tz), side="left")
    return np.asfortranarray(out)
