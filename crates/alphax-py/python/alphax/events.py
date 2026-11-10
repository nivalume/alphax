"""Event studies: what the price does around a signal.

Every event is re-anchored to its own bar, so a breakout in March and one in
October line up on one offset axis and can be averaged::

    study = alphax.events.forward_paths(close, entries, before=5, after=20)
    study.mean_path()      # typical drift, offset by offset
    study.median_path()    # the same without the outliers
    study.hit_rate(20)     # how often it was up 20 bars later

A strong mean carried by one lucky event looks very different from a weak one
that almost always works; the three together tell them apart.

**This reads the future.** The value at offset ``k > 0`` is a price that did
not exist when the event fired. Use a study to decide *whether* a signal is
worth simulating, never as an input to the simulation.
"""

from __future__ import annotations

from typing import Any

from . import _alphax
from ._columns import columns
from ._alphax import EventPaths

__all__ = ["EventPaths", "forward_paths"]


def forward_paths(close: Any, events: Any, *, after: int, before: int = 0) -> EventPaths:
    """The return path around every bar where ``events`` is true.

    Parameters
    ----------
    close
        Prices, ``(bars,)`` or ``(bars, columns)``.
    events
        Booleans shaped like ``close``. Floats are refused: a NaN cast to bool
        is ``True``, which would put an event on every missing bar.
    after, before
        How many bars after and before each event to keep. The offset axis is
        ``-before..=after``.

    Returns
    -------
    EventPaths
        ``paths`` is ``(n_events, before + after + 1)``, each row
        ``close[t + k] / close[t] - 1``; ``col`` and ``idx`` say where each
        event came from, column by column and in bar order within one. An
        event too close to either end of its column is kept, padded with
        NaN — dropping it would bias the study toward the middle of the
        sample — and every reduction skips NaN.
    """
    close_arr, _ = columns(close, "close")
    events_arr, _ = columns(events, "events", bool)
    return _alphax.forward_paths(close_arr, events_arr, before, after)
