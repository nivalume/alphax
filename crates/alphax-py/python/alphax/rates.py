"""Turning annual rates into the per-bar rates the engine charges.

The engine's holding charges — ``debt_rate``, ``borrow_rate`` and the
financing rates — take what one bar's holding costs, for the time until the
next bar, because the engine has no clock: a bar is an index. How much time a
bar covers is in the bars' timestamps, which only the caller has::

    idx = df.index                                # the bars' timestamps
    debt = alphax.rates.per_bar(0.06, idx)        # 6% a year, per bar
    pf = alphax.from_orders(close, size, leverage=2.0, debt_rate=debt)

A daily stock series charges Friday's bar for three days, the weekend
included, and a bar before a holiday for the holiday too; a crypto series that
trades around the clock charges every bar alike. The last bar covers no time
and charges nothing: the run ends at its close.

Funding needs no conversion. An exchange publishes the rate each funding time
settles at; put it on the bar whose close is that time and leave the other bars
``NaN``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["per_bar"]

_NS_PER_DAY = 86_400 * 10**9


def _timestamps(index: Any) -> np.ndarray:
    """The bars' times as int64 nanoseconds."""
    if getattr(index, "tz", None) is not None:
        # A zoned pandas index, as UTC: what counts is the time between bars,
        # which a daylight-saving change would otherwise distort.
        index = index.tz_convert(None)
    times = np.asarray(index)
    if times.dtype.kind != "M":
        raise TypeError(
            "index must hold datetimes: a pandas DatetimeIndex or a datetime64 array; "
            f"got dtype {times.dtype}"
        )
    if times.ndim != 1:
        raise ValueError(f"index must be one-dimensional; got {times.ndim} dimensions")
    times = times.astype("datetime64[ns]")
    if np.isnat(times).any():
        raise ValueError("index cannot contain NaT")
    return times.view(np.int64)


def per_bar(annual_rate: Any, index: Any, *, days_per_year: float = 365.0) -> np.ndarray:
    """An annual rate as the rate each bar charges, from the bars' timestamps.

    ``rate[t] = annual_rate[t] * days(index[t + 1] - index[t]) / days_per_year``,
    and the last bar is 0.

    Parameters
    ----------
    annual_rate
        A number, one value per bar ``(n_bars,)``, or a matrix
        ``(n_bars, n_cols)``. A ``(1, n_cols)`` row gives each column its own
        rate. ``NaN`` stays ``NaN``, which charges nothing.
    index
        The bars' timestamps, strictly increasing: a pandas ``DatetimeIndex``
        or a ``datetime64`` array.
    days_per_year
        The day count: 365 (the default) for crypto and sterling, 360 for most
        money-market rates in dollars and euros.

    Returns
    -------
    numpy.ndarray
        ``(n_bars,)`` for a number or one value per bar, ``(n_bars, n_cols)``
        for a matrix.
    """
    if not np.isfinite(days_per_year) or days_per_year <= 0:
        raise ValueError(f"days_per_year must be positive; got {days_per_year}")
    times = _timestamps(index)
    steps = np.diff(times)
    if (steps <= 0).any():
        raise ValueError("index must be strictly increasing")
    days = np.append(steps / _NS_PER_DAY, 0.0)
    rate = np.asarray(annual_rate, dtype=np.float64)
    if rate.ndim == 0:
        return float(rate) * days / days_per_year
    if rate.ndim == 1:
        if rate.shape[0] != days.shape[0]:
            raise ValueError(
                f"a 1-D annual_rate is one value per bar: {days.shape[0]} bars, "
                f"got {rate.shape[0]} values"
            )
        return rate * days / days_per_year
    if rate.ndim == 2:
        if rate.shape[0] not in (1, days.shape[0]):
            raise ValueError(
                f"a 2-D annual_rate has one row per bar or a single row: {days.shape[0]} "
                f"bars, got {rate.shape[0]} rows"
            )
        return np.asfortranarray(rate * days[:, None] / days_per_year)
    raise ValueError(f"annual_rate must be a number, 1-D or 2-D; got {rate.ndim} dimensions")
