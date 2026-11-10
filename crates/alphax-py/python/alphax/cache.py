"""Remember results across calls and runs, and count how often they are reused.

A sweep or a notebook often recomputes the same backtest or factor set on the
same inputs. :class:`Cache` keys a call by a digest of its function and
arguments (NumPy arrays by their bytes, shape and dtype), keeps recent
results in memory and, with a directory, on disk, so a later process finds
them too::

    cache = alphax.cache.Cache("~/.cache/alphax", max_items=64)

    @cache.memoize
    def run(window):
        return alphax.from_signals(close, *signals(window), size=1.0)

    run(20); run(20)
    cache.stats()   # {"hits": 1, "misses": 1, "hit_rate": 0.5, ...}

Measure the hit rate on the real workload before relying on a cache: one
that never hits only adds hashing and storage. Portfolios are stored as their
versioned bytes (``Portfolio.to_bytes``), anything else with pickle. Inputs
must not change in place between calls with the same object: the digest is
of the contents at call time, so a change gives a new key, but a function
that reads globals is keyed only by its arguments and its own code.
"""

from __future__ import annotations

import functools
import hashlib
import os
import pickle
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Cache", "digest"]


def _feed(h: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        h.update(b"nd")
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, (list, tuple)):
        h.update(b"seq" if isinstance(value, list) else b"tup")
        h.update(str(len(value)).encode())
        for v in value:
            _feed(h, v)
    elif isinstance(value, dict):
        h.update(b"map")
        for k in sorted(value, key=repr):
            _feed(h, k)
            _feed(h, value[k])
    elif callable(value) and hasattr(value, "__qualname__"):
        h.update(f"fn:{getattr(value, '__module__', '')}.{value.__qualname__}".encode())
        # Its code too, so an edited function does not find its old results;
        # what the code reads from globals or closures is not seen.
        code = getattr(value, "__code__", None)
        if code is not None:
            h.update(code.co_code)
            _feed(h, [c for c in code.co_consts if not hasattr(c, "co_code")])
    elif hasattr(value, "to_bytes") and not isinstance(value, int):
        h.update(b"bytes")
        h.update(value.to_bytes())
    else:
        h.update(repr(value).encode())
    h.update(b"|")


def digest(*parts: Any) -> str:
    """A hex digest of the values, stable across processes."""
    h = hashlib.blake2b(digest_size=20)
    for p in parts:
        _feed(h, p)
    return h.hexdigest()


class Cache:
    """An in-memory LRU of ``max_items`` results, optionally backed by a
    directory."""

    def __init__(self, directory: str | os.PathLike | None = None, max_items: int = 128):
        self.directory = None if directory is None else Path(directory).expanduser()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self._memory: OrderedDict[str, Any] = OrderedDict()
        self.hits = self.misses = self.disk_hits = 0

    def _path(self, key: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{key}.pkl"

    def get(self, key: str) -> tuple[bool, Any]:
        if key in self._memory:
            self._memory.move_to_end(key)
            self.hits += 1
            return True, self._memory[key]
        if self.directory is not None and self._path(key).exists():
            with open(self._path(key), "rb") as f:
                kind, payload = pickle.load(f)
            value = _decode(kind, payload)
            self._remember(key, value)
            self.hits += 1
            self.disk_hits += 1
            return True, value
        self.misses += 1
        return False, None

    def put(self, key: str, value: Any) -> None:
        self._remember(key, value)
        if self.directory is not None:
            tmp = self._path(key).with_suffix(".tmp")
            with open(tmp, "wb") as f:
                pickle.dump(_encode(value), f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._path(key))

    def _remember(self, key: str, value: Any) -> None:
        self._memory[key] = value
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_items:
            self._memory.popitem(last=False)

    def memoize(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """``fn`` with its results cached by its arguments."""

        @functools.wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            key = digest(fn, args, kwargs)
            found, value = self.get(key)
            if found:
                return value
            value = fn(*args, **kwargs)
            self.put(key, value)
            return value

        wrapped.cache = self  # type: ignore[attr-defined]
        return wrapped

    def stats(self) -> dict[str, Any]:
        """Calls served from the cache (``hits``, of which ``disk_hits``
        came from the directory), calls computed (``misses``), and
        ``hit_rate``."""
        total = self.hits + self.misses
        return {"hits": self.hits, "disk_hits": self.disk_hits, "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0, "in_memory": len(self._memory)}

    def clear(self) -> None:
        """Forgets everything, on disk too, and resets the counts."""
        self._memory.clear()
        if self.directory is not None:
            for p in self.directory.glob("*.pkl"):
                p.unlink()
        self.hits = self.misses = self.disk_hits = 0


def _encode(value: Any) -> tuple[str, Any]:
    from . import Portfolio

    if isinstance(value, Portfolio):
        return "portfolio", value.to_bytes()
    return "pickle", value


def _decode(kind: str, payload: Any) -> Any:
    from . import Portfolio

    return Portfolio.from_bytes(payload) if kind == "portfolio" else payload
