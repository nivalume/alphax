"""A `from_orders` run advanced one bar at a time, for live trading.

The strategy decides a bar's orders; ``propose`` says what the backtest
would fill. An execution layer places them and reports what actually filled;
``commit`` books those fills, so the records are what happened rather than
what was simulated, and ``reconcile`` compares positions with the broker's.

Each bar runs alone, from the last bar's state, on the bar and the
``lookback`` bars before it, so a bar costs the same however long the
history: see ``alphax.SimState`` and ``from_orders(resume=, row_offset=)``.
Committing every proposal as it stands writes the records one
``from_orders`` run over the whole history writes.

::

    st = alphax.Stepper(n_cols=2, init_cash=10_000.0, fees=0.001)
    for bar in feed:
        proposal = st.propose(bar.close, size=decide(bar))
        fills = broker.execute(proposal)         # per column: size, price, fees
        st.commit(fills)                         # or st.commit() to take the proposal
        drift = st.reconcile(broker.positions())
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Stepper"]

# Parameters that make an order, replaced when booking a reported fill: the
# fill is the order, at its price and with its fee, and the account has to
# hold it.
_FILL_ORDER = {
    "size_type": "amount",
    "direction": "both",
    "fees": 0.0,
    "slippage": 0.0,
    "min_size": 0.0,
    "max_size": np.inf,
    "size_granularity": np.nan,
    "cash_limit": np.inf,
    "reject_prob": 0.0,
    "lock_cash": False,
    "allow_partial": False,
    "raise_reject": True,
}

# Per-bar inputs a bar may carry, shaped like close; the lookback bars carry
# them too, with the market's own and no order.
_MARKET = ("open", "high", "low", "mark_price", "funding_rate", "long_financing_rate",
           "short_financing_rate", "borrow_rate", "cash_dividends", "bid_price", "ask_price")


class Stepper:
    """Advances a ``from_orders`` run bar by bar, booking reported fills.

    ``n_cols`` columns; every other keyword is a ``from_orders`` argument
    that holds for every bar (``init_cash``, ``fees``, ``leverage``,
    ``account_type``, ``group_lens``, ``cash_sharing``...). Per-bar inputs go
    to ``propose``. ``lookback`` is how many bars before each the run reads:
    1 covers ``price=-inf`` (the previous close); ``from_ago`` is not
    supported, since the decision is the bar's own.
    """

    def __init__(self, n_cols: int, *, lookback: int = 1, **params: Any):
        from . import from_orders  # noqa: F401 (import cycle with the package)

        for name in ("size", "close", "resume", "row_offset", "from_ago"):
            if name in params:
                raise TypeError(f"{name} is per bar or the stepper's; pass it to propose()")
        if lookback < 0:
            raise ValueError("lookback must be 0 or more")
        self.n_cols = int(n_cols)
        self.lookback = int(lookback)
        self.params = params
        self.state = None
        self.row = 0
        self._history: dict[str, list[np.ndarray]] = {"close": []}
        self._pending = None
        self._orders: list[dict] = []
        self._cash: list[dict] = []

    def _row(self, value, name, dtype=float):
        v = np.asarray(value, dtype=dtype)
        if v.ndim == 0:
            v = np.full(self.n_cols, v, dtype=dtype)
        if v.shape != (self.n_cols,):
            raise ValueError(f"{name} must be a number or {self.n_cols} values; got {v.shape}")
        return v

    def _run(self, bar: dict, size: np.ndarray, extra: dict):
        from . import from_orders

        n_back = len(self._history["close"])
        def window(name, current, fill):
            past = self._history.get(name)
            rows = list(past) if past is not None and len(past) == n_back else [fill] * n_back
            return np.asfortranarray(np.vstack(rows + [current]))

        nan_row = np.full(self.n_cols, np.nan)
        args = dict(self.params)
        args.update(extra)
        for name, current in bar.items():
            if name == "close":
                continue
            args[name] = window(name, current, nan_row)
        out = from_orders(
            window("close", bar["close"], nan_row),
            window("size", size, nan_row),
            resume=self.state,
            row_offset=self.row - n_back,
            **args,
        )
        orders = out.orders()
        cash = out.cash_records()
        return orders, cash, out.last_state

    def propose(self, close, size=np.nan, **bar: Any) -> dict:
        """What the backtest would fill on the next bar: the run's
        ``orders()`` for it, nothing committed.

        ``close`` and ``size`` are one value per column (or one for all);
        ``bar`` holds the bar's other inputs: ``price``, ``open``, ``high``,
        ``low``, rates. A ``NaN`` size is no order.
        """
        current = {"close": self._row(close, "close")}
        for name, value in bar.items():
            if name in ("resume", "row_offset", "from_ago"):
                raise TypeError(f"{name} is the stepper's")
            current[name] = self._row(value, name)
        size = self._row(size, "size")
        orders, cash, state = self._run(current, size, {})
        self._pending = (current, size, orders, cash, state)
        return orders

    def commit(self, fills: dict | None = None) -> dict:
        """Books the proposed bar and moves to the next.

        With ``fills`` ``None`` the proposal is booked as it stands. Otherwise
        ``fills`` is what the broker reported: ``size`` (signed, per column;
        ``NaN`` or 0 for none), ``price`` and ``fees`` (cash, per column).
        Several fills of one column on one bar are reported as one: their
        total size, average price and total fee. The bar's cash flows and
        charges are booked either way. Returns the fills booked.

        A reported fill the account cannot hold (not enough cash, a size the
        column's direction or margin refuses) raises: alphax's account and the
        broker's no longer agree, and ``reconcile`` says where.
        """
        if self._pending is None:
            raise RuntimeError("propose() a bar before committing it")
        current, _, orders, cash, state = self._pending
        if fills is not None:
            size = self._row(fills.get("size", np.nan), "size")
            price = self._row(fills.get("price", np.nan), "price")
            fees = self._row(fills.get("fees", 0.0), "fees")
            size = np.where(size == 0, np.nan, size)
            bar = {k: v for k, v in current.items() if k != "price"}
            bar["price"] = price
            extra = dict(_FILL_ORDER, fixed_fees=np.nan_to_num(fees))
            orders, cash, state = self._run(bar, size, extra)
        self._orders.append(orders)
        self._cash.append(cash)
        self.state = state
        for name, value in current.items():
            past = self._history.setdefault(name, [])
            if len(past) != len(self._history["close"]):
                past[:] = [np.full(self.n_cols, np.nan)] * len(self._history["close"])
            past.append(value)
        for past in self._history.values():
            del past[: max(0, len(past) - self.lookback)]
        self.row += 1
        self._pending = None
        return orders

    @property
    def position(self) -> np.ndarray:
        """Each column's position after the last committed bar."""
        if self.state is None:
            return np.zeros(self.n_cols)
        return np.asarray(self.state.position)

    @property
    def cash(self) -> np.ndarray:
        """Each account group's cash after the last committed bar."""
        if self.state is None:
            raise RuntimeError("no bar committed yet")
        return np.asarray(self.state.cash)

    def reconcile(self, positions) -> np.ndarray:
        """The broker's positions less alphax's, per column: zero where they
        agree."""
        return self._row(positions, "positions") - self.position

    def _group_of(self) -> np.ndarray:
        lens = self.params.get("group_lens")
        if lens is None or not self.params.get("cash_sharing", False):
            return np.arange(self.n_cols)
        return np.repeat(np.arange(len(lens)), lens)

    def orders(self) -> dict:
        """Every committed fill, as ``Portfolio.orders()`` gives them, on the
        history's rows, in the order one run over the history writes them:
        by account group, then by bar."""
        return _concat(self._orders, self._group_of())

    def cash_records(self) -> dict:
        """Every committed cash flow, as ``Portfolio.cash_records()``."""
        return _concat(self._cash, self._group_of())


def _concat(parts: list[dict], group_of: np.ndarray) -> dict:
    if not parts:
        return {}
    keys = parts[-1].keys()
    out = {k: np.concatenate([p[k] for p in parts if k in p]) for k in keys}
    order = np.argsort(group_of[out["col"]], kind="stable")
    return {k: v[order] for k, v in out.items()}
