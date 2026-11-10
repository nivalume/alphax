"""Developer helpers: what a function takes, and how inputs will broadcast."""

from __future__ import annotations

import inspect
import textwrap
from typing import Any

import numpy as np

__all__ = ["check_shapes", "help"]

# Market data beside the close: a 1-D array is one column, like close.
_MARKET = {"open", "high", "low", "close", "price", "val_price", "mark_price", "bid_price", "ask_price",
           "cash_dividends", "funding_rate", "long_financing_rate", "short_financing_rate", "borrow_rate",
           "entries", "exits", "long_entries", "long_exits", "short_entries", "short_exits"}
# Parameters whose 1-D form is one value per bar.
_PER_ROW = {"cash_deposits", "cash_earnings", "debt_rate", "force_exit", "auto_deleverage", "index"}


def _resolve(obj: Any) -> Any:
    if not isinstance(obj, str):
        return obj
    import alphax

    target: Any = alphax
    for part in obj.split("."):
        if not hasattr(target, part):
            raise AttributeError(f"alphax has no {obj!r}")
        target = getattr(target, part)
    return target


def help(obj: Any, *, full: bool = False, echo: bool = True) -> str:
    """A short reference for ``obj`` (an object or a dotted name under
    ``alphax``, e.g. ``"indicators.rsi"``): its signature, the first
    paragraph of its docstring (all of it with ``full``) and, for a module
    or class, its public members. Printed (unless ``echo=False``) and
    returned."""
    target = _resolve(obj)
    name = getattr(target, "__qualname__", getattr(target, "__name__", repr(target)))
    lines = []
    try:
        lines.append(f"{name}{inspect.signature(target)}")
    except (TypeError, ValueError):
        lines.append(name)
    doc = inspect.getdoc(target) or "(no docstring)"
    if not full:
        doc = doc.split("\n\n")[0]
    lines.append(textwrap.indent(doc, "    "))
    if inspect.ismodule(target) or inspect.isclass(target):
        members = getattr(target, "__all__", None) or [m for m in dir(target) if not m.startswith("_")]
        if members:
            lines.append("    members: " + ", ".join(sorted(members)))
    text = "\n".join(lines)
    if echo:
        print(text)
    return text


def check_shapes(close: Any, **inputs: Any) -> dict[str, str]:
    """How each input would broadcast against ``close``, before a run:
    ``scalar``, ``per bar``, ``per column``, ``full``, or the error the
    run would raise. Raises ``ValueError`` listing every bad input.

    Checks plain runs: with ``column_map`` the run's columns are the map's,
    and shared ``cash_deposits`` are ``(bars, groups)``; neither is
    modelled here."""
    c = np.asarray(close)
    n_rows = c.shape[0]
    n_cols = 1 if c.ndim == 1 else c.shape[1]
    out, bad = {}, []
    for name, value in inputs.items():
        if value is None:
            out[name] = "default"
            continue
        a = np.asarray(value)
        if a.ndim == 0:
            out[name] = "scalar"
        elif a.ndim == 1:
            if name in _MARKET:
                ok = len(a) == n_rows and n_cols == 1
                out[name] = "full (one column)" if ok else f"error: a 1-D {name} is one column of {n_rows} bars; got {len(a)} values for {n_cols} columns"
            elif name in _PER_ROW:
                out[name] = "per bar" if len(a) == n_rows else f"error: {len(a)} values for {n_rows} bars"
            else:
                out[name] = "per column" if len(a) == n_cols else f"error: {len(a)} values for {n_cols} columns (a 1-D parameter is one value per column)"
        elif a.ndim == 2:
            out[name] = "full" if a.shape == (n_rows, n_cols) else f"error: a {a.shape[0]}x{a.shape[1]} array against a {n_rows}x{n_cols} run"
        else:
            out[name] = f"error: {a.ndim} dimensions"
        if out[name].startswith("error"):
            bad.append(f"{name}: {out[name][7:]}")
    if bad:
        raise ValueError("inputs that would not broadcast:\n  " + "\n  ".join(bad))
    return out
