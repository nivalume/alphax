"""Shape plumbing shared by the research modules.

Each of them takes "a series or a panel" and hands back the same: a 1-D input
is one column and comes back 1-D, a 2-D input is ``(bars, columns)`` and comes
back 2-D. The engine reads column-major, so everything goes in F-ordered —
free when it already is, a visible copy when it is not.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def columns(value: Any, name: str, dtype: Any = np.float64) -> tuple[np.ndarray, bool]:
    """``value`` as a 2-D F-ordered array, and whether it came in 1-D.

    A boolean input must already be boolean. Casting floats to bool would turn
    every NaN into ``True`` — an event on every missing bar — so it is refused
    instead of guessed at.
    """
    array = np.asarray(value)
    if dtype is bool:
        if array.dtype != np.dtype(bool):
            raise TypeError(f"{name} must contain booleans; got {array.dtype}")
    else:
        array = array.astype(dtype, copy=False)
    if array.ndim == 1:
        return np.asfortranarray(array.reshape(-1, 1)), True
    if array.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D; got {array.ndim} dimensions")
    return np.asfortranarray(array), False


def restore(array: np.ndarray, one_d: bool) -> np.ndarray:
    """Undo :func:`columns`'s reshape on a result of the same shape."""
    return array[:, 0] if one_d else array


def per_column(values: np.ndarray, one_d: bool) -> Any:
    """One value per column, or the single value of a 1-D input."""
    return float(values[0]) if one_d else values


def shape(value: Any) -> tuple[int, int, bool]:
    """``(n_rows, n_cols, one_d)`` from ``n`` or ``(n_rows, n_cols)``.

    An int asks for one series and gets a 1-D array, as NumPy's own
    generators do.
    """
    if isinstance(value, (int, np.integer)):
        return int(value), 1, True
    dims = tuple(int(d) for d in value)
    if len(dims) != 2:
        raise ValueError(f"shape must be n or (n_rows, n_cols); got {value!r}")
    return dims[0], dims[1], False
