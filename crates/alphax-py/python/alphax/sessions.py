"""Trading sessions: which bars are inside one, and which end one.

An intraday strategy flattens before the close and does not trade outside
the session. The engine has no clock, so the session is a mask built here
from the bars' timestamps, in the exchange's time zone:

::

    inside = alphax.sessions.session_mask(times, "09:30", "16:00", tz="America/New_York")
    last = alphax.sessions.session_end(times, "09:30", "16:00", tz="America/New_York")
    pf = alphax.from_signals(close, entries & inside[:, None], exits, force_exit=last)

``force_exit`` closes the position on the session's last bar in place of
anything else ordered there. A bar is in the session when its timestamp's
local time of day is in ``[start, end)``; a session that ends before it
starts (``"18:00"`` to ``"17:00"``) runs overnight, and belongs to the day
it starts on. Which instant a bar's timestamp marks — its open or its close
— is the caller's: pass the one the session is meant to bound.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .rates import _timestamps

__all__ = ["session_end", "session_mask"]

_NS = 1_000_000_000
_DAY = 86_400 * _NS
_HOUR = 3_600 * _NS


def _clock(value: Any, name: str) -> int:
    """A time of day as nanoseconds after midnight."""
    if isinstance(value, str):
        try:
            value = _dt.time.fromisoformat(value)
        except ValueError as e:
            raise ValueError(f"{name} must be a time of day such as '09:30'; got {value!r}") from e
    if not isinstance(value, _dt.time):
        raise TypeError(f"{name} must be a 'HH:MM' string or a datetime.time; got {value!r}")
    seconds = value.hour * 3600 + value.minute * 60 + value.second
    return (seconds * 1_000_000 + value.microsecond) * 1000


def _local(times: np.ndarray, tz: Any) -> np.ndarray:
    """UTC nanoseconds as the same instants on the local wall clock.

    The offset is looked up once per distinct UTC hour, which every
    daylight-saving change falls on.
    """
    if tz is None:
        return times
    zone = ZoneInfo(tz) if isinstance(tz, str) else tz
    hours = times // _HOUR
    distinct, which = np.unique(hours, return_inverse=True)
    offsets = np.array(
        [
            int(
                _dt.datetime.fromtimestamp(int(h) * 3600, tz=_dt.timezone.utc)
                .astimezone(zone)
                .utcoffset()
                .total_seconds()
            )
            * _NS
            for h in distinct
        ],
        dtype=np.int64,
    )
    return times + offsets[which]


def _session(index: Any, start: Any, end: Any, tz: Any, weekdays: Any) -> tuple[np.ndarray, np.ndarray]:
    """Whether each bar is in a session, and which session (its start day)."""
    times = _timestamps(index)
    local = _local(times, tz)
    open_at, close_at = _clock(start, "start"), _clock(end, "end")
    if open_at == close_at:
        raise ValueError("start and end must differ")
    since_open = local - open_at
    day = np.floor_divide(since_open, _DAY)
    into = since_open - day * _DAY
    length = (close_at - open_at) % _DAY
    inside = into < length
    # 1970-01-01, day 0, was a Thursday; Monday is 0.
    weekday = (day + 3) % 7
    inside &= np.isin(weekday, np.asarray(list(weekdays)))
    return inside, day


def session_mask(
    index: Any,
    start: Any,
    end: Any,
    *,
    tz: Any = None,
    weekdays: Any = (0, 1, 2, 3, 4),
) -> np.ndarray:
    """Whether each bar's time is inside a session.

    Parameters
    ----------
    index
        The bars' times: a pandas ``DatetimeIndex`` or a ``datetime64``
        array, as UTC when not zoned.
    start, end
        The session's local times of day, ``"HH:MM"`` or ``datetime.time``.
    tz
        The exchange's time zone, an IANA name or a ``tzinfo``; ``None`` is
        UTC.
    weekdays
        The days sessions start on, Monday 0; Monday to Friday by default.
    """
    inside, _ = _session(index, start, end, tz, weekdays)
    return inside


def session_end(
    index: Any,
    start: Any,
    end: Any,
    *,
    tz: Any = None,
    weekdays: Any = (0, 1, 2, 3, 4),
) -> np.ndarray:
    """Whether each bar is the last of its session: a mask for ``force_exit``.

    The last bar inside a session, whichever bar it is — a session cut short
    by missing data ends on the last bar it has. Arguments as
    :func:`session_mask`.
    """
    inside, day = _session(index, start, end, tz, weekdays)
    last = inside.copy()
    following = np.zeros_like(inside)
    following[:-1] = inside[1:] & (day[1:] == day[:-1])
    last &= ~following
    return last
