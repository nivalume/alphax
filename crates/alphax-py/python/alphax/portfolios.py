"""Portfolios cut, joined and resampled, by rebuilding them from records.

A portfolio is its fills and cash flows, the closes they are valued at,
the starting cash and the grouping; ``Portfolio.from_records`` builds one
from those. Everything here transforms them and builds again, so the result
analyses as any other portfolio. Logs are not carried over.

- ``select(pf, accounts)``: some accounts (groups when cash is shared,
  columns otherwise), in the order given.
- ``column_stack(*pfs)``: portfolios over the same bars, side by side.
- ``row_slice(pf, start, stop)``: the bars ``start:stop``, starting from
  the account as it stood before ``start``: its cash, and its positions as
  initial positions bought at the close before ``start``. Returns, value and
  drawdowns over those bars are the original's; trades are cut at ``start``.
  Not for runs that borrow or trade on margin.
- ``resample(pf, bar_of)``: onto coarser bars; ``bar_of`` gives each bar's
  coarse bar (``resample_bars(index, "W")`` makes it from timestamps). Fills
  keep their prices, and are valued at each coarse bar's last close.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._alphax import Portfolio

__all__ = ["column_stack", "resample", "resample_bars", "row_slice", "select"]


def _starts(lens: Sequence[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)


def _source(pf: Portfolio) -> np.ndarray:
    """Each run column's close column."""
    m = pf.column_map
    return np.arange(pf.close.shape[1]) if m is None else np.asarray(m)


def _rebuild(pf: Portfolio, orders: dict, cash: dict, close: np.ndarray, *, init_cash, group_lens,
             column_map=None) -> Portfolio:
    s = pf.scoring
    shared = pf.cash_sharing
    return Portfolio.from_records(
        orders,
        np.asfortranarray(close),
        init_cash=np.asarray(init_cash, dtype=np.float64),
        cash_records=cash,
        group_lens=list(map(int, group_lens)) if shared else None,
        cash_sharing=shared,
        column_map=None if column_map is None else list(map(int, column_map)),
        allow_negative_price=s["allow_negative_price"],
        ann_factor=s["ann_factor"],
        risk_free=s["risk_free"],
        trades=s["trades"],
        incl_open=s["incl_open"],
    )


def _group_init_cash(pf: Portfolio) -> np.ndarray:
    cash = np.asarray(pf.init_cash)
    if not pf.cash_sharing:
        return cash
    return cash[_starts(pf.group_lens)]


def _take(records: dict, keep: np.ndarray) -> dict:
    return {k: np.asarray(v)[keep] for k, v in records.items()}


def select(pf: Portfolio, accounts: Sequence[int]) -> Portfolio:
    """The accounts ``accounts`` of ``pf``, in that order."""
    lens = np.asarray(pf.group_lens if pf.cash_sharing else np.ones(len(_source(pf)), dtype=int))
    accounts = [int(a) for a in accounts]
    if any(a < 0 or a >= len(lens) for a in accounts):
        raise IndexError(f"accounts must be in 0..{len(lens)}")
    starts = _starts(lens)
    cols = np.concatenate([np.arange(starts[a], starts[a] + lens[a]) for a in accounts]).astype(np.int64)
    new_of = np.full(int(lens.sum()), -1)
    new_of[cols] = np.arange(len(cols))
    group_of_new = np.repeat(np.arange(len(accounts)), [lens[a] for a in accounts])

    def remap(records: dict) -> dict:
        keep = new_of[np.asarray(records["col"], dtype=np.int64)] >= 0
        out = _take(records, keep)
        out["col"] = new_of[out["col"]]
        # One run writes by group, then as written: keep that order.
        order = np.argsort(group_of_new[out["col"]], kind="stable")
        return {k: v[order] for k, v in out.items()}

    source = _source(pf)[cols]
    used, column_map = np.unique(source, return_inverse=True)
    init = _group_init_cash(pf)[accounts]
    return _rebuild(pf, remap(pf.orders()), remap(pf.cash_records()), pf.close[:, used],
                    init_cash=init, group_lens=[lens[a] for a in accounts], column_map=column_map)


def column_stack(*pfs: Portfolio) -> Portfolio:
    """Portfolios over the same bars, side by side, as one."""
    if not pfs:
        raise ValueError("nothing to stack")
    first = pfs[0]
    if any(p.close.shape[0] != first.close.shape[0] for p in pfs):
        raise ValueError("portfolios must cover the same bars")
    if any(p.scoring != first.scoring or p.cash_sharing != first.cash_sharing for p in pfs):
        raise ValueError("portfolios must share their scoring and whether they share cash")
    orders, cash, closes, maps, inits, lens = [], [], [], [], [], []
    col_off = src_off = 0
    for p in pfs:
        n_cols = len(_source(p))
        o, c = p.orders(), p.cash_records()
        o["col"] = np.asarray(o["col"]) + col_off
        c["col"] = np.asarray(c["col"]) + col_off
        orders.append(o)
        cash.append(c)
        closes.append(p.close)
        maps.append(_source(p) + src_off)
        inits.append(_group_init_cash(p))
        lens.extend(p.group_lens if p.cash_sharing else [1] * n_cols)
        col_off += n_cols
        src_off += p.close.shape[1]
    keys = set.intersection(*(set(o) for o in orders))
    if len(keys) != len(orders[0]) or any(set(o) != keys for o in orders):
        raise ValueError("portfolios must all keep balances with their fills, or none")
    o = {k: np.concatenate([x[k] for x in orders]) for k in orders[0]}
    c = {k: np.concatenate([x[k] for x in cash]) for k in cash[0]}
    return _rebuild(first, o, c, np.hstack(closes), init_cash=np.concatenate(inits), group_lens=lens,
                    column_map=np.concatenate(maps))


def row_slice(pf: Portfolio, start: int, stop: int | None = None) -> Portfolio:
    """The bars ``start:stop`` of ``pf``, continuing from where it stood."""
    n_rows = pf.close.shape[0]
    stop = n_rows if stop is None else int(stop)
    start = int(start)
    if not 0 <= start < stop <= n_rows:
        raise IndexError(f"need 0 <= start < stop <= {n_rows}; got {start}:{stop}")
    orders, cash = pf.orders(), pf.cash_records()
    if "debt" in orders:
        raise NotImplementedError("row_slice does not cut runs that borrow or trade on margin")
    source = _source(pf)
    n_cols = len(source)
    idx = np.asarray(orders["idx"])
    keep = (idx >= start) & (idx < stop)
    body = _take(orders, keep)
    body["idx"] = body["idx"] - start
    ckeep = (np.asarray(cash["idx"]) >= start) & (np.asarray(cash["idx"]) < stop)
    cbody = _take(cash, ckeep)
    cbody["idx"] = cbody["idx"] - start
    if start == 0:
        return _rebuild(pf, body, cbody, pf.close[:stop], init_cash=_group_init_cash(pf),
                        group_lens=pf.group_lens, column_map=pf.column_map)
    # Positions and cash at the end of the bar before.
    before = idx < start
    signed = np.where(np.asarray(orders["side"]) == 0, 1.0, -1.0) * np.asarray(orders["size"])
    position = np.bincount(np.asarray(orders["col"])[before], weights=signed[before], minlength=n_cols)
    price = pf.close[start - 1, source]
    accounts = len(pf.group_lens) if pf.cash_sharing else n_cols
    cash_then = np.array([(pf.cash(group=g) if pf.cash_sharing else pf.cash(g))[start - 1]
                          for g in range(accounts)])
    open_cols = np.flatnonzero(position != 0)
    init_fills = {
        "col": open_cols.astype(np.int64),
        "idx": np.zeros(len(open_cols), dtype=np.int64),
        "size": np.abs(position[open_cols]),
        "price": price[open_cols],
        "fees": np.zeros(len(open_cols)),
        "side": np.where(position[open_cols] > 0, 0, 1).astype(np.int64),
    }
    init_cash_flows = {
        "col": open_cols.astype(np.int64),
        "idx": np.zeros(len(open_cols), dtype=np.int64),
        "amount": position[open_cols] * price[open_cols],
        "kind": np.full(len(open_cols), 8, dtype=np.int64),
    }
    o = {k: np.concatenate([init_fills[k], body[k]]) for k in body}
    c = {k: np.concatenate([init_cash_flows[k], cbody[k]]) for k in cbody}
    group_of = np.repeat(np.arange(accounts), pf.group_lens) if pf.cash_sharing else np.arange(n_cols)
    for rec in (o, c):
        order = np.argsort(group_of[rec["col"]], kind="stable")
        for k in rec:
            rec[k] = rec[k][order]
    return _rebuild(pf, o, c, pf.close[start:stop], init_cash=cash_then, group_lens=pf.group_lens,
                    column_map=pf.column_map)


def resample_bars(index: Any, freq: str) -> np.ndarray:
    """Each bar's coarse bar for ``freq`` (``"D"``, ``"W"``, ``"ME"``,
    ``"QE"``, ``"YE"``): 0 for the bars of the first period, then 1..."""
    from .signals import calendar

    first = calendar(index, freq, which="first")
    return np.cumsum(first) - 1


def resample(pf: Portfolio, bar_of: Any) -> Portfolio:
    """``pf`` on coarser bars: ``bar_of[i]`` is bar ``i``'s coarse bar,
    non-decreasing from 0."""
    bar_of = np.asarray(bar_of, dtype=np.int64)
    n_rows = pf.close.shape[0]
    if bar_of.shape != (n_rows,) or (bar_of[0] != 0 if n_rows else False) or (np.diff(bar_of) < 0).any() \
            or (np.diff(bar_of) > 1).any():
        raise ValueError("bar_of must give each bar its coarse bar: from 0, rising by 0 or 1")
    last = np.flatnonzero(np.diff(np.append(bar_of, bar_of[-1] + 1)))
    close = pf.close[last]
    orders, cash = pf.orders(), pf.cash_records()
    orders["idx"] = bar_of[np.asarray(orders["idx"], dtype=np.int64)]
    cash["idx"] = bar_of[np.asarray(cash["idx"], dtype=np.int64)]
    return _rebuild(pf, orders, cash, close, init_cash=_group_init_cash(pf), group_lens=pf.group_lens,
                    column_map=pf.column_map)
