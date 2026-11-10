"""Small Python conveniences for signal strategies."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from inspect import signature
from typing import Any, Protocol, cast

import numpy as np

from . import FactorSet, _alphax, from_signals
from .data import BarPanel


class StrategyFunction(Protocol):
    def __call__(self, data: Any, **params: Any) -> Any: ...

    def run(self, data: Any, *, params: Mapping[str, Any] | None = None, signal_delay: int = 1, **options: Any) -> Any: ...

    def sweep(self, data: Any, *, params: Any, signal_delay: int = 1, **options: Any) -> Any: ...


def _field(data, name):
    if isinstance(data, Mapping):
        if name not in data:
            raise ValueError(f"market data is missing field {name!r}")
        return data[name]
    if isinstance(data, BarPanel):
        if not hasattr(data, name):
            raise ValueError(f"market data is missing field {name!r}")
        return getattr(data, name)
    raise TypeError("market data must be a BarPanel or a mapping of fields to arrays")


def factors(data, **expressions):
    """Evaluate named expressions against the fields they require, with
    warm-up: each factor is ``NaN`` on a symbol's rows before its windows
    have filled (:meth:`FactorSet.with_warmup`), so no signal comes from a
    partial window. ``FactorSet.compile(...).evaluate(...)`` keeps Qlib's
    values from the first row."""
    compiled = FactorSet.compile(expressions).with_warmup()
    if compiled.uses_future:
        raise ValueError("factor expressions cannot use future data")
    names = compiled.required_fields or ("close",)
    return compiled.evaluate({name: _field(data, name) for name in names})


def crossed_above(left, right):
    """Cross from below to above, using the engine's equality/NaN semantics."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim not in (1, 2):
        raise ValueError("crossed_above inputs must have the same 1-D or 2-D shape")
    if a.ndim == 1:
        a, b = a[:, None], b[:, None]
    if 0 in a.shape:
        return np.zeros(a.shape, dtype=bool)
    flat = _alphax.crossed_above(np.asfortranarray(a), np.asfortranarray(b))
    return flat.reshape(a.shape, order="F")


def _signal(value, name, shape):
    array = np.asarray(value)
    if array.dtype != np.dtype(bool):
        raise TypeError(f"{name} must contain booleans")
    if array.ndim == 1 and shape[1] == 1:
        array = array[:, None]
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
    return array


def _delay(array, bars):
    if bars == 0:
        return array
    delayed = np.zeros_like(array)
    if bars < len(array):
        delayed[bars:] = array[:-bars]
    return delayed


def strategy(func: Any) -> StrategyFunction:
    """Add run/sweep methods to a signal-producing function."""
    if not callable(func):
        raise TypeError("strategy requires a callable")
    strategy_signature = signature(func)

    @wraps(func)
    def wrapped(*args, **kwargs):
        return func(*args, **kwargs)

    def sweep(data, *, params, signal_delay=1, **options):
        if (
            isinstance(signal_delay, (bool, np.bool_))
            or not isinstance(signal_delay, (int, np.integer))
            or signal_delay < 0
        ):
            raise ValueError("signal_delay must be a nonnegative integer")
        managed = ("close", "entries", "exits", "open", "high", "low", "column_map", "group_lens")
        if any(name in options for name in managed):
            raise TypeError("market data, signals, column_map and group_lens are managed by strategy")
        close = np.asarray(_field(data, "close"), dtype=np.float64)
        if close.ndim == 1:
            close = close[:, None]
        if close.ndim != 2:
            raise ValueError("close must be 1-D or 2-D")
        try:
            groups = list(params)
        except TypeError as exc:
            raise TypeError("params must be a nonempty sequence of mappings") from exc
        if not groups:
            raise ValueError("params cannot be empty")

        entries, exits = [], []
        for group in groups:
            if not isinstance(group, Mapping):
                raise TypeError("each parameter set must be a mapping")
            strategy_signature.bind(data, **group)
            signals = wrapped(data, **group)
            if not isinstance(signals, tuple) or len(signals) != 2:
                raise TypeError("strategy must return an (entries, exits) tuple")
            entries.append(_delay(_signal(signals[0], "entries", close.shape), signal_delay))
            exits.append(_delay(_signal(signals[1], "exits", close.shape), signal_delay))

        market = {
            name: _field(data, name)
            for name in ("open", "high", "low")
            if (isinstance(data, Mapping) and name in data)
            or (isinstance(data, BarPanel) and hasattr(data, name))
        }
        sharing = options.get("cash_sharing", False)
        if sharing:
            options["group_lens"] = [close.shape[1]] * len(groups)
        return from_signals(
            close,
            np.concatenate(entries, axis=1),
            np.concatenate(exits, axis=1),
            column_map=np.tile(np.arange(close.shape[1]), len(groups)),
            **market,
            **options,
        )

    def run(data, *, params=None, signal_delay=1, **options):
        return sweep(
            data, params=[{} if params is None else params],
            signal_delay=signal_delay, **options,
        )

    managed = cast(StrategyFunction, wrapped)
    managed.run = run
    managed.sweep = sweep
    return managed
