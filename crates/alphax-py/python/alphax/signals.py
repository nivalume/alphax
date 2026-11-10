"""Signals: random baselines, cleaning, generated exits, ranges, ranks,
calendar signals and a summary table.

A strategy whose entries beat :func:`random_n` entries with the same count
has an entry rule; one that does not has a sample.

Every draw is a function of ``(seed, row, col)`` alone, not of a random
stream: the same seed gives the same signals, bit for bit, and a column's
signals do not change when other columns are generated beside it. They are
not NumPy's draws and do not try to be.

``shape`` is ``n`` for one 1-D series or ``(n_rows, n_cols)`` for a panel.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _alphax
from ._columns import columns as _columns, restore, shape as _shape

__all__ = [
    "calendar",
    "clean",
    "exit_after",
    "exit_on",
    "generate_ohlc_stop_exits",
    "normalized_average_index",
    "nth_index",
    "random_by_prob",
    "random_entries_exits",
    "random_like",
    "random_n",
    "ranges",
    "rank",
    "stats",
]


def random_by_prob(shape: Any, prob: float, *, seed: int = 0) -> np.ndarray:
    """Each bar fires independently with probability ``prob``.

    The count per column is binomial, not fixed; use :func:`random_n` when it
    must match another signal's.
    """
    n_rows, n_cols, one_d = _shape(shape)
    return restore(_alphax.random_by_prob(n_rows, n_cols, float(prob), int(seed)), one_d)


def random_n(shape: Any, n: int, *, seed: int = 0) -> np.ndarray:
    """Exactly ``n`` signals per column, at distinct rows drawn uniformly."""
    n_rows, n_cols, one_d = _shape(shape)
    return restore(_alphax.random_n(n_rows, n_cols, int(n), int(seed)), one_d)


def random_entries_exits(shape: Any, n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """``n`` entry/exit pairs per column, strictly alternating.

    ``2n`` distinct rows are drawn, sorted, and dealt entry, exit, entry,
    exit — so every entry has its exit before the next entry and no bar is
    both. Returns ``(entries, exits)``.
    """
    n_rows, n_cols, one_d = _shape(shape)
    entries, exits = _alphax.random_entries_exits(n_rows, n_cols, int(n), int(seed))
    return restore(entries, one_d), restore(exits, one_d)


def _stop(value: Any, name: str) -> Any:
    """A stop input: a number, one value per column (1-D) or a matrix."""
    if value is None or isinstance(value, (int, float)):
        return value
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return np.ascontiguousarray(array)
    if array.ndim == 2:
        return np.asfortranarray(array)
    raise ValueError(f"{name} must be a number, a 1-D array or a 2-D array")


def generate_ohlc_stop_exits(
    entries: Any,
    close: Any,
    *,
    open: Any = None,
    high: Any = None,
    low: Any = None,
    sl_stop: Any = None,
    sl_trail: bool = False,
    tsl_stop: Any = None,
    tsl_th: Any = None,
    tp_stop: Any = None,
    td_stop: Any = None,
    dt_stop: Any = None,
    short: bool = False,
    wait: int = 1,
    until_next: bool = True,
    skip_until_exit: bool = False,
    delta_format: str = "percent",
    stop_ladder: str = "disabled",
    stop_order_type: str = "market",
    stop_limit_delta: Any = None,
    limit_order_price: str = "limit",
    out: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each entry's stops would fire, without running a simulation.

    Each entry is taken at its bar's close and its stops — fractions of that
    close, NaN off; ``td_stop`` in bars after the entry, ``dt_stop`` a row —
    are tested on the bars after it with the simulation's own test. Returns
    the exits, the price each fired at (NaN elsewhere) and which stop fired
    (0 stop-loss, 1 trailing, 2 take-profit, 3 time, -1 elsewhere), shaped
    like ``close``.

    ``until_next`` ends an entry's search at the next entry; ``skip_until_exit``
    ignores the entries before an exit. With ``until_next=False`` and
    ``skip_until_exit=True`` the exits are the stop exits ``from_signals``
    fills for the same entries when it ignores an entry while holding. ``wait``
    is how many bars after the entry the first test is, at least 1.

    ``delta_format`` reads the stop distances as ``from_signals`` does
    (``"percent"``, ``"percent100"``, ``"absolute"``, ``"target"``).
    ``stop_ladder`` makes the rows of ``sl_stop``, ``tsl_stop`` and
    ``tp_stop`` a ladder's steps: one exit per step, each closing part of the
    position. ``stop_order_type="limit"`` places a stop-limit order where a
    stop fires, ``stop_limit_delta`` worse, and the exit is where it fills
    (``limit_order_price`` as in ``from_signals``); not with a ladder.
    Pass a dict as ``out`` to receive ``out["size"]``: the share of the
    entry's position each exit closes (1, or a ladder step's part).
    """
    close_2d, one_d = _columns(close, "close")
    entries_2d, _ = _columns(entries, "entries", dtype=bool)
    bar = lambda a, name: None if a is None else _columns(a, name)[0]  # noqa: E731
    exits, price, kind, size = _alphax.generate_ohlc_stop_exits(
        entries_2d, close_2d, open=bar(open, "open"), high=bar(high, "high"),
        low=bar(low, "low"), sl_stop=_stop(sl_stop, "sl_stop"), sl_trail=bool(sl_trail),
        tsl_stop=_stop(tsl_stop, "tsl_stop"), tsl_th=_stop(tsl_th, "tsl_th"),
        tp_stop=_stop(tp_stop, "tp_stop"), td_stop=_stop(td_stop, "td_stop"),
        dt_stop=_stop(dt_stop, "dt_stop"), short=bool(short), wait=int(wait),
        until_next=bool(until_next), skip_until_exit=bool(skip_until_exit),
        delta_format=delta_format, stop_ladder=stop_ladder, stop_order_type=stop_order_type,
        stop_limit_delta=_stop(stop_limit_delta, "stop_limit_delta"),
        limit_order_price=limit_order_price,
    )
    if out is not None:
        out["size"] = restore(size, one_d)
    return restore(exits, one_d), restore(price, one_d), restore(kind, one_d)


def _bool(value: Any, name: str) -> tuple[np.ndarray, bool]:
    return _columns(value, name, bool)


def clean(entries: Any, exits: Any, *, entry_first: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Entries and exits made to alternate: an entry, then the next exit,
    then the next entry. A bar with both keeps neither. With
    ``entry_first=False`` a column may also start with an exit."""
    en, one_d = _bool(entries, "entries")
    ex, _ = _bool(exits, "exits")
    a, b = _alphax.sig_clean_enex(en, ex, bool(entry_first))
    return restore(a, one_d), restore(b, one_d)


def exit_after(entries: Any, bars: int, *, until: str = "next_entry") -> np.ndarray:
    """An exit ``bars`` bars after each entry.

    ``until`` bounds the search: ``"next_entry"`` (an exit belongs to one
    entry; none past the next entry), ``"end_skip"`` (to the end, skipping
    entries inside a resolved window) or ``"end"`` (every entry on its own).
    """
    en, one_d = _bool(entries, "entries")
    return restore(_alphax.sig_exit_after(en, int(bars), until), one_d)


def exit_on(entries: Any, condition: Any, *, wait: int = 1, until: str = "next_entry") -> np.ndarray:
    """An exit on the first bar where ``condition`` holds, at least ``wait``
    bars after each entry; ``until`` as for :func:`exit_after`."""
    en, one_d = _bool(entries, "entries")
    cond, _ = _bool(condition, "condition")
    return restore(_alphax.sig_exit_on(en, cond, int(wait), until), one_d)


def ranges(signals: Any, kind: str = "partition") -> dict:
    """The signal's spans as records: ``col``, ``start_idx``, ``end_idx``,
    ``status`` (0 open, 1 closed) and ``duration``.

    ``kind`` is ``"partition"`` (each run of ``True``), ``"between"`` (from one
    signal to the next) or ``"between_partition"`` (from the end of one run to
    the start of the next).
    """
    a, _ = _bool(signals, "signals")
    return _alphax.sig_ranges(a, kind)


def rank(signals: Any, *, reset_by: Any = None, partition: bool = False, allow_gaps: bool = False,
         skip_leading: bool = False) -> np.ndarray:
    """Each signal's rank in its run, from 0 (-1 where there is none).

    ``partition`` ranks the runs instead (every bar of the second run is 1);
    ``allow_gaps`` keeps counting across the gaps between runs; ``reset_by``
    restarts the count wherever it is ``True``; ``skip_leading`` leaves a run
    already under way at the first bar unranked.
    """
    a, one_d = _bool(signals, "signals")
    reset = None if reset_by is None else _bool(reset_by, "reset_by")[0]
    return restore(_alphax.sig_rank(a, reset, bool(partition), bool(allow_gaps), bool(skip_leading)), one_d)


def nth_index(signals: Any, n: int) -> Any:
    """Each column's ``n``th signal row (from 0; negative from the end), -1
    where there is none."""
    a, one_d = _bool(signals, "signals")
    out = _alphax.sig_nth_index(a, int(n))
    return int(out[0]) if one_d else out


def normalized_average_index(signals: Any) -> Any:
    """Where each column's signals sit on average, from -1 (all at the
    start) to 1 (all at the end). Signals bunched at one end usually mean a
    warm-up or look-ahead problem."""
    a, one_d = _bool(signals, "signals")
    out = _alphax.sig_normalized_average_index(a)
    return float(out[0]) if one_d else out


_PERIODS = {"D": "D", "W": "W", "ME": "M", "M": "M", "QE": "Q", "Q": "Q", "YE": "Y", "Y": "Y"}


def calendar(index: Any, freq: str = "ME", *, which: str = "first", n: int = 0) -> np.ndarray:
    """A signal on the bars that open (``which="first"``) or close
    (``"last"``) each calendar period of ``index``: ``"D"``, ``"W"`` (weeks
    from Monday), ``"ME"`` (months), ``"QE"`` (quarters) or ``"YE"``.

    ``n`` counts from there: ``which="first", n=2`` is each month's third
    bar. Only the bars there are count, so a month's first trading day is
    its first bar, whatever the date.
    """
    from .rates import _timestamps

    if freq not in _PERIODS:
        raise ValueError(f"freq must be one of {sorted(_PERIODS)}; got {freq!r}")
    if which not in ("first", "last"):
        raise ValueError("which must be 'first' or 'last'")
    times = _timestamps(index).view("datetime64[ns]")
    unit = _PERIODS[freq]
    if unit == "W":
        # Weeks from Monday: 1970-01-01 was a Thursday.
        days = times.astype("datetime64[D]").astype(np.int64)
        period = (days + 3) // 7
    elif unit == "Q":
        months = times.astype("datetime64[M]").astype(np.int64)
        period = months // 3
    else:
        period = times.astype(f"datetime64[{unit}]").astype(np.int64)
    if (np.diff(period) < 0).any():
        raise ValueError("index must be sorted")
    change = np.flatnonzero(np.diff(period)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(period)]])
    out = np.zeros(len(period), dtype=bool)
    rows = starts + n if which == "first" else ends - 1 - n
    keep = (rows >= starts) & (rows < ends)
    out[rows[keep]] = True
    return out


def stats(signals: Any) -> dict:
    """A signal's summary per column: how many, when the first and last
    fire, the rate per bar, the gaps between them and the runs they form
    (mean, median, 10th and 90th percentiles), and where they sit on
    average (:func:`normalized_average_index`). Each value is an array with
    one entry per column."""
    a, _ = _bool(signals, "signals")
    n_rows, n_cols = a.shape
    out: dict[str, np.ndarray] = {
        "count": a.sum(axis=0),
        "rate": a.mean(axis=0) if n_rows else np.full(n_cols, np.nan),
        "first_index": _alphax.sig_nth_index(a, 0),
        "last_index": _alphax.sig_nth_index(a, -1),
        "normalized_average_index": _alphax.sig_normalized_average_index(a),
    }
    for name, kind in (("gap", "between"), ("run", "partition")):
        r = _alphax.sig_ranges(a, kind)
        closed = r["status"] == 1 if kind == "partition" else np.ones(len(r["col"]), dtype=bool)
        for stat, f in (("mean", np.mean), ("median", np.median),
                        ("p10", lambda v: np.percentile(v, 10)), ("p90", lambda v: np.percentile(v, 90))):
            vals = np.full(n_cols, np.nan)
            for c in range(n_cols):
                d = r["duration"][(r["col"] == c) & closed]
                if len(d):
                    vals[c] = f(d.astype(float))
            out[f"{name}_{stat}"] = vals
    return out


def random_like(signals: Any, *, seed: int = 0, keep: str = "count") -> np.ndarray:
    """A random baseline matched to ``signals``, column by column.

    ``keep="count"`` keeps each column's number of signals and draws their
    rows uniformly (as :func:`random_n`). ``keep="gaps"`` keeps the gaps
    between them too, in a random order, and a random start, so the baseline
    trades as often and as regularly as the strategy, only at other bars.
    """
    a, one_d = _bool(signals, "signals")
    n_rows, n_cols = a.shape
    out = np.zeros_like(a)
    for c in range(n_cols):
        rows = np.flatnonzero(a[:, c])
        if keep == "count":
            if len(rows):
                out[:, c] = _alphax.random_n(n_rows, 1, len(rows), int(seed) * 1_000_003 + c)[:, 0]
            continue
        if keep != "gaps":
            raise ValueError("keep must be 'count' or 'gaps'")
        if len(rows) == 0:
            continue
        # Seeded by (seed, column): the same column gets the same baseline
        # whatever else is generated beside it.
        rng = np.random.default_rng([int(seed), c])
        gaps = np.diff(rows)
        gaps = gaps[rng.permutation(len(gaps))]
        span = int(gaps.sum())
        start = int(rng.integers(0, n_rows - span)) if n_rows > span else 0
        out[np.concatenate([[start], start + np.cumsum(gaps)]), c] = True
    return restore(out, one_d)
