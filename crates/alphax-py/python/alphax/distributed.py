"""Backtests split by column across processes or a Ray cluster.

Accounts never interact, so a run over many columns can be cut into groups
of whole accounts, run anywhere, and stacked back; the result is the run
over all columns, bit for bit::

    pf = alphax.distributed.run("from_signals", close, entries, exits,
                                n_chunks=8, backend="process", size=1.0)
    pf = alphax.distributed.run("from_orders", close, size, backend="ray")

Every array argument with one value per column of ``close`` (2-D, or a
1-D per-column array) is cut to the chunk's columns; others are passed
whole. A 1-D array as long as the bars is read per bar, and passed whole,
unless there are exactly as many bars as columns: then pass it 2-D. With ``cash_sharing``, chunks hold whole groups of
``group_lens``. ``backend`` is ``"process"`` (a ``ProcessPoolExecutor``),
``"thread"``, ``"ray"`` (needs ``ray``, started with ``ray.init()``
or here), or ``"serial"``. Results are stacked with
``alphax.portfolios.column_stack``, which rebuilds from records: leverage
and margin accounts are not supported, and a result stacked from several
chunks has no ``last_state`` to continue with ``resume=``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any

import numpy as np

__all__ = ["chunks", "run"]


def chunks(group_lens: list[int], n_chunks: int) -> list[tuple[int, int, int, int]]:
    """``(first_group, last_group, first_col, last_col)`` per chunk: groups
    split into ``n_chunks`` runs of about equal column count."""
    lens = np.asarray(group_lens, dtype=np.int64)
    ends = np.cumsum(lens)
    total = int(ends[-1]) if len(ends) else 0
    n_chunks = max(1, min(n_chunks, len(lens)))
    out, g0 = [], 0
    for k in range(1, n_chunks + 1):
        target = total * k / n_chunks
        g1 = int(np.searchsorted(ends, target, side="left")) + 1 if k < n_chunks else len(lens)
        g1 = max(g1, g0 + 1)
        if g0 >= len(lens):
            break
        g1 = min(g1, len(lens))
        c0 = int(ends[g0 - 1]) if g0 else 0
        out.append((g0, g1, c0, int(ends[g1 - 1])))
        g0 = g1
    return out


def _cut(value: Any, n_rows: int, n_cols: int, c0: int, c1: int) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == n_cols and n_cols > 1:
        return value[:, c0:c1]
    if isinstance(value, np.ndarray) and value.ndim == 1 and n_rows != n_cols and len(value) == n_cols and n_cols > 1:
        return value[c0:c1]
    return value


def _job(entry: str, close: np.ndarray, args: tuple, kwargs: dict, as_bytes: bool = False) -> Any:
    import alphax

    pf = getattr(alphax, entry)(close, *args, **kwargs)
    # A portfolio crosses a process boundary as its versioned bytes.
    return pf.to_bytes() if as_bytes else pf


def _portfolio(data: bytes) -> Any:
    import alphax

    return alphax.Portfolio.from_bytes(data)


def run(entry: str, close: Any, *args: Any, n_chunks: int = 4, backend: str = "process",
        **kwargs: Any) -> Any:
    """``alphax.<entry>(close, *args, **kwargs)`` over column chunks, stacked."""
    from . import portfolios

    if entry not in ("from_signals", "from_orders"):
        raise ValueError("entry must be 'from_signals' or 'from_orders'")
    if "column_map" in kwargs:
        raise TypeError("column_map is not supported here; cut the columns before")
    c = np.asarray(close, dtype=float)
    c = c[:, None] if c.ndim == 1 else c
    n_rows, n_cols = c.shape
    sharing = bool(kwargs.get("cash_sharing", False))
    lens = list(kwargs.get("group_lens") or ([n_cols] if sharing else [1] * n_cols))
    if not sharing:
        lens = [1] * n_cols
    args = tuple(np.asarray(a) if isinstance(a, (list, tuple)) else a for a in args)
    jobs = []
    for g0, g1, c0, c1 in chunks(lens, n_chunks):
        kw = {k: _cut(v, n_rows, n_cols, c0, c1) for k, v in kwargs.items()}
        if sharing:
            kw["group_lens"] = lens[g0:g1]
        init = kw.get("init_cash")
        if isinstance(init, np.ndarray) and init.ndim == 1 and len(init) == len(lens) and len(lens) != n_cols:
            kw["init_cash"] = init[g0:g1]
        # Shared deposits are one column per group.
        dep = kwargs.get("cash_deposits")
        if (sharing and isinstance(dep, np.ndarray) and dep.ndim == 2 and dep.shape[1] == len(lens)
                and len(lens) != n_cols):
            kw["cash_deposits"] = dep[:, g0:g1]
        jobs.append((entry, c[:, c0:c1], tuple(_cut(a, n_rows, n_cols, c0, c1) for a in args), kw))
    if backend == "serial":
        parts = [_job(*j) for j in jobs]
    elif backend in ("process", "thread"):
        remote = backend == "process"
        pool = ProcessPoolExecutor if remote else ThreadPoolExecutor
        with pool(max_workers=len(jobs)) as ex:
            parts = list(ex.map(_job, *zip(*jobs), [remote] * len(jobs)))
        if remote:
            parts = [_portfolio(b) for b in parts]
    elif backend == "ray":
        import ray

        if not ray.is_initialized():
            ray.init()
        task = ray.remote(_job)
        parts = [_portfolio(b) for b in ray.get([task.remote(*j, True) for j in jobs])]
    else:
        raise ValueError("backend must be 'process', 'thread', 'ray' or 'serial'")
    return parts[0] if len(parts) == 1 else portfolios.column_stack(*parts)
