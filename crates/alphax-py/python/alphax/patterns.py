"""Pattern search: where a series traced a given shape.

``search(close, pattern)`` slides a window along each column, rescales the
window and the pattern alike (min-max to 0..1 by default, so only the shape
counts, not the level or the size of the move), and scores each window by
``1 - mean |window - pattern|``. Windows scoring at least ``min_similarity``
are matches; overlapping matches give way to the best of them unless
``overlap=True``. ``window`` may be one length or several, to find the
shape drawn faster or slower: the pattern is stretched to each by linear
interpolation.

::

    head_and_shoulders = [0, 0.6, 0.3, 1.0, 0.3, 0.6, 0]
    hits = alphax.patterns.search(close, head_and_shoulders, window=[20, 40], min_similarity=0.85)
    hits["start_idx"], hits["end_idx"], hits["similarity"]

Each match ends on ``end_idx - 1``: a signal built from it may act from
``end_idx`` on, after the shape is complete, never before.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._columns import columns

__all__ = ["search", "signals"]


def _rescale(a: np.ndarray, how: str | None) -> np.ndarray:
    if how is None:
        return a
    if how == "minmax":
        lo = np.nanmin(a, axis=-1, keepdims=True)
        span = np.nanmax(a, axis=-1, keepdims=True) - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(span > 0, (a - lo) / span, 0.5)
    if how == "zscore":
        mu = np.nanmean(a, axis=-1, keepdims=True)
        sd = np.nanstd(a, axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, (a - mu) / sd, 0.0)
    raise ValueError("rescale must be 'minmax', 'zscore' or None")


def _stretch(pattern: np.ndarray, n: int) -> np.ndarray:
    if len(pattern) == n:
        return pattern
    return np.interp(np.linspace(0, len(pattern) - 1, n), np.arange(len(pattern)), pattern)


def search(series: Any, pattern: Sequence[float], *, window: int | Sequence[int] | None = None,
           min_similarity: float = 0.9, rescale: str | None = "minmax", overlap: bool = False) -> dict:
    """The windows of each column of ``series`` that trace ``pattern``, as
    records: ``col``, ``start_idx``, ``end_idx`` (exclusive), ``window`` and
    ``similarity``, ordered by column and start."""
    s, _ = columns(series, "series")
    pattern = np.asarray(pattern, dtype=np.float64)
    if pattern.ndim != 1 or len(pattern) < 2:
        raise ValueError("pattern must be at least two values")
    windows = [len(pattern)] if window is None else ([window] if np.isscalar(window) else list(window))
    n_rows, n_cols = s.shape
    found = []
    for w in windows:
        w = int(w)
        if w < 2 or w > n_rows:
            continue
        p = _rescale(_stretch(pattern, w), rescale)
        views = np.lib.stride_tricks.sliding_window_view(s, w, axis=0)  # (n_rows - w + 1, n_cols, w)
        scaled = _rescale(views, rescale)
        err = np.abs(scaled - p).mean(axis=-1)
        if rescale == "zscore":
            err = err / 2.0
        sim = 1.0 - err
        sim = np.where(np.isnan(views).any(axis=-1), np.nan, sim)
        rows, cols = np.nonzero(sim >= min_similarity)
        for r, c in zip(rows, cols):
            found.append((int(c), int(r), int(r) + w, w, float(sim[r, c])))
    if not overlap:
        kept = []
        taken: dict[int, list[tuple[int, int]]] = {}
        for c, a, b, w, v in sorted(found, key=lambda t: -t[4]):
            if any(a < e and s_ < b for s_, e in taken.get(c, [])):
                continue
            taken.setdefault(c, []).append((a, b))
            kept.append((c, a, b, w, v))
        found = kept
    found.sort(key=lambda t: (t[0], t[1]))
    arr = np.array(found, dtype=object).reshape(-1, 5)
    return {
        "col": arr[:, 0].astype(np.int64),
        "start_idx": arr[:, 1].astype(np.int64),
        "end_idx": arr[:, 2].astype(np.int64),
        "window": arr[:, 3].astype(np.int64),
        "similarity": arr[:, 4].astype(np.float64),
    }


def signals(matches: dict, shape: tuple[int, int]) -> np.ndarray:
    """A signal on the first bar after each match, ``end_idx``, where the
    shape is complete; ``shape`` is the series'."""
    out = np.zeros(shape, dtype=bool)
    ok = matches["end_idx"] < shape[0]
    out[matches["end_idx"][ok], matches["col"][ok]] = True
    return out
