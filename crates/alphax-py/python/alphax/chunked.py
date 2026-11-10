"""Long backtests in row chunks, with checkpoints to resume after a stop.

``run`` splits the bars into chunks and runs them one after another, each
continuing the previous one's state (``resume=state, row_offset=``), so the
records are bit for bit those of one run over all bars. After each chunk it
can write a checkpoint (the state and the records so far); called again with
the same checkpoint it skips the chunks already done, which also extends a
finished run when the history has grown (the bars it ran must be unchanged)::

    pf = alphax.chunked.run("from_signals", close, entries, exits,
                            chunk=50_000, checkpoint="run.ckpt", size=1.0, fees=0.001)

Every array argument with one row per bar is cut to the chunk, plus
``lookback`` bars before it for anything that reads past bars (a price
``from_ago``, the previous close an order price may read). 2-D arrays with
one row per bar are cut; so are 1-D ones when ``close`` is 1-D, and the
1-D arguments read per bar (``index``, ``force_exit``, ``auto_deleverage``,
``cash_deposits``, ``cash_earnings``, ``debt_rate``) always.

The result is rebuilt from the records with ``Portfolio.from_records``, so
it carries what analysis reads from records: values, returns, trades,
statistics. Like ``portfolios.row_slice``, that rebuild does not cover
leverage or margin accounts, and the result has no ``last_state``: continue
a chunked run by running ``chunked.run`` again over the longer history with
the same checkpoint, which picks up after the last finished chunk. Library code never writes files; the
checkpoint is written here, in Python.
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["run"]

# 1-D arguments read one value per bar (the rest are one value per column).
_PER_ROW_1D = {"index", "force_exit", "auto_deleverage", "cash_deposits", "cash_earnings", "debt_rate"}
_ENTRY_POINTS = ("from_signals", "from_orders")


def _cut(value: Any, name: str, n_rows: int, one_column: bool, lo: int, hi: int) -> Any:
    if isinstance(value, np.ndarray) or (isinstance(value, (list, tuple)) and len(value) == n_rows):
        a = np.asarray(value)
        if a.ndim == 2 and a.shape[0] == n_rows and n_rows > 1:
            return a[lo:hi]
        if a.ndim == 1 and len(a) == n_rows and (one_column or name in _PER_ROW_1D):
            return a[lo:hi]
    return value


def _concat(parts: list[Mapping[str, Any]], group_of: np.ndarray) -> dict[str, np.ndarray]:
    """The chunks' records in the order of one run: by account, and in time
    order within it (a stable sort of chunk order), as ``OrderRecords::chain``
    orders them."""
    parts = [p for p in parts if p]
    if not parts:
        return {}
    keys = parts[0].keys()
    out = {k: np.concatenate([np.asarray(p[k]) for p in parts]) for k in keys}
    if "col" in out and len(out["col"]):
        order = np.argsort(group_of[np.asarray(out["col"], dtype=np.intp)], kind="stable")
        out = {k: v[order] for k, v in out.items()}
    return out


def _save(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def run(
    entry: str,
    close: Any,
    *args: Any,
    chunk: int,
    lookback: int = 1,
    checkpoint: str | os.PathLike | None = None,
    **kwargs: Any,
) -> Any:
    """Runs ``alphax.<entry>(close, *args, **kwargs)`` in chunks of
    ``chunk`` bars and returns the portfolio of the whole run.

    ``entry`` is ``"from_signals"`` or ``"from_orders"``. ``checkpoint`` is
    a file path: written after each chunk, read at the start if it exists and
    was made by a run of the same shape and chunking.
    """
    import alphax

    if entry not in _ENTRY_POINTS:
        raise ValueError(f"entry must be one of {_ENTRY_POINTS}")
    if chunk < 1 or lookback < 0:
        raise ValueError("chunk must be at least 1 and lookback at least 0")
    for name in ("resume", "row_offset"):
        if name in kwargs:
            raise TypeError(f"{name} is managed by chunked.run")
    if isinstance(kwargs.get("init_cash"), str):
        raise ValueError("init_cash='auto' needs the whole run; give a number")
    fn = getattr(alphax, entry)
    close_a = np.asarray(close, dtype=float)
    n_rows = close_a.shape[0]
    one_column = close_a.ndim == 1
    from .cache import digest

    # A checkpoint belongs to a run: the entry point, the columns and the
    # chunking, plus a digest of every input on the bars it finished. A call
    # with a longer history whose first bars and other arguments digest the
    # same continues it.
    signature = (entry, close_a.shape[1:], chunk, lookback)

    def prefix(rows: int) -> str:
        return digest(close_a[:rows], [_cut(a, "", n_rows, one_column, 0, rows) for a in args],
                      {k: _cut(v, k, n_rows, one_column, 0, rows) for k, v in kwargs.items()})

    done, state, orders, cash = 0, None, [], []
    if checkpoint is not None and os.path.exists(checkpoint):
        with open(checkpoint, "rb") as f:
            saved = pickle.load(f)
        if saved["signature"] != signature or saved["done"] > n_rows or saved["inputs"] != prefix(saved["done"]):
            raise ValueError(f"checkpoint {checkpoint} is from another run: {saved['signature']}, "
                             f"{saved['done']} bars, or other inputs on them")
        done, orders, cash = saved["done"], saved["orders"], saved["cash"]
        state = alphax.SimState.from_bytes(saved["state"]) if saved["state"] is not None else None

    first_pf = None
    for start in range(done, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        lo = 0 if state is None else max(0, start - lookback)
        cut_args = [_cut(a, "", n_rows, one_column, lo, stop) for a in args]
        cut_kwargs = {k: _cut(v, k, n_rows, one_column, lo, stop) for k, v in kwargs.items()}
        if state is None:
            result = fn(close_a[lo:stop], *cut_args, **cut_kwargs)
            first_pf = result
        else:
            result = fn(close_a[lo:stop], *cut_args, resume=state, row_offset=lo, **cut_kwargs)
        orders.append(dict(result.orders()))
        cash.append(dict(result.cash_records()))
        state = result.last_state
        if checkpoint is not None:
            _save(os.fspath(checkpoint), {"signature": signature, "done": stop, "inputs": prefix(stop),
                                          "state": state.to_bytes(), "orders": orders, "cash": cash})

    template = first_pf if first_pf is not None else fn(close_a[:1], *[_cut(a, "", n_rows, one_column, 0, 1) for a in args],
                                                       **{k: _cut(v, k, n_rows, one_column, 0, 1) for k, v in kwargs.items()})
    lens = list(template.group_lens) if template.cash_sharing else []
    n_cols = len(template.column_map) if template.column_map is not None else (1 if one_column else close_a.shape[1])
    group_of = np.repeat(np.arange(len(lens)), lens) if lens else np.arange(n_cols)
    # `init_cash` reads per column (a group's cash repeated over its columns);
    # `from_records` takes one value per account, so a shared run hands over
    # each group's first column.
    from .portfolios import _group_init_cash

    return alphax.Portfolio.from_records(
        _concat(orders, group_of), np.asfortranarray(close_a if not one_column else close_a[:, None]),
        init_cash=_group_init_cash(template), cash_records=_concat(cash, group_of),
        group_lens=template.group_lens if template.cash_sharing else None,
        cash_sharing=template.cash_sharing, column_map=template.column_map,
        allow_negative_price=bool(kwargs.get("allow_negative_price", False)),
        **{k: kwargs[k] for k in ("ann_factor", "risk_free", "trades", "incl_open") if k in kwargs},
    )
