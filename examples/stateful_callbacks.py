"""Stateful Path-Dependent Callbacks: `from_order_func` & `from_signals` Hooks.

Scenario & Dimensions Covered:
  1. `from_order_func` with Full Lifecycle Hooks & Custom State (`pf.in_outputs`):
     - `pre_group_func`: Initializes a per-group mutable state dict stored in `pf.in_outputs`.
     - `pre_segment_func`: Evaluates regime/cooldown filters and returns `"skip"` when
       no trading is needed on the bar.
     - `order_func`: Computes path-dependent order sizes based on live `cash`, `position`,
       and custom state (e.g. martingale / anti-martingale win-streak sizing).
     - `post_order_func`: Inspects the exact `fill` dictionary (`size`, `price`, `fees`,
       `side`, `direction`) immediately after execution to update entry cost & streak state.
     - `post_segment_func`: Monitors intra-simulation drawdown and returns `"stop"` to
       halt trading for the group if a circuit breaker is breached.
  2. Flexible Multi-Leg Pair Trading (`from_order_func(..., flexible=True)`):
     - Returning a list of `(col, order_spec)` tuples from a single callback to execute
       simultaneous long/short spread legs in a shared cash pool (`cash_sharing=True`).
  3. Row-Wise Cross-Group Coordination (`from_order_func(..., row_wise=True)`):
     - Iterating bar-by-bar across multiple independent groups.
  4. `from_signals` Path-Dependent Hooks (`signal_func` and `adjust_func`):
     - `signal_func`: Generating entries/exits dynamically from live account equity & position.
     - `adjust_func`: Ratcheting stop-loss and take-profit levels dynamically as unrealized
       profit grows on an open position.

Run:
    uv run python examples/stateful_callbacks.py
"""

from __future__ import annotations

import numpy as np
import alphax


def demo_order_func_lifecycle_and_circuit_breaker() -> None:
    print("=== 1. `from_order_func` Full Lifecycle Hooks & Drawdown Circuit Breaker ===")
    rng = np.random.default_rng(12)
    n_bars = 180
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.016, size=(n_bars, 1)), axis=0))
    fast = alphax.indicators.ma(close, 6).ma
    slow = alphax.indicators.ma(close, 18).ma

    def pre_group(group: int, from_col: int, to_col: int) -> dict:
        _ = (group, from_col, to_col)
        return {
            "peak_value": 10_000.0,
            "entry_price": np.nan,
            "win_streak": 0,
            "halted_at_bar": None,
            "fills_logged": 0,
        }

    def pre_segment(ctx: alphax.Context, state: dict) -> str | None:
        if ctx.row < 18:
            return "skip"  # skip indicator warm-up bars
        return None

    def order_cb(ctx: alphax.Context, k: int, state: dict) -> dict | float | None:
        i = ctx.row
        pos = ctx.position[k]
        price = close[i, k]
        if pos == 0.0 and fast[i, k] > slow[i, k]:
            # Scale position size between 20% and 50% of cash based on consecutive winning trades
            alloc_pct = min(0.20 + 0.10 * state["win_streak"], 0.50)
            qty = (ctx.cash * alloc_pct) / price
            return {"size": qty, "size_type": "amount", "direction": "longonly"}
        if pos > 0.0 and fast[i, k] < slow[i, k]:
            return {"size": 0.0, "size_type": "targetamount"}
        return None

    def post_order(ctx: alphax.Context, k: int, result: dict, state: dict) -> None:
        _ = (ctx, k)
        if not result["filled"]:
            return
        state["fills_logged"] += 1
        if result["side"] == 0:  # 0 = Buy, 1 = Sell
            state["entry_price"] = result["price"]
        elif result["side"] == 1 and not np.isnan(state["entry_price"]):
            if result["price"] > state["entry_price"]:
                state["win_streak"] += 1
            else:
                state["win_streak"] = 0
            state["entry_price"] = np.nan

    def post_segment(ctx: alphax.Context, state: dict) -> str | None:
        val = ctx.value
        state["peak_value"] = max(state["peak_value"], val)
        dd = (state["peak_value"] - val) / state["peak_value"]
        if dd > 0.08:  # 8% drawdown circuit breaker
            state["halted_at_bar"] = ctx.row
            return "stop"
        return None

    pf = alphax.from_order_func(
        close,
        order_cb,
        pre_group_func=pre_group,
        pre_segment_func=pre_segment,
        post_order_func=post_order,
        post_segment_func=post_segment,
        init_cash=10_000.0,
        fees=0.0005,
    )
    st = pf.in_outputs[0]
    print(
        f"Return: {pf.total_return()[0]:.2%} | Fills logged in state: {st['fills_logged']} | "
        f"Final win_streak: {st['win_streak']} | Circuit breaker bar: {st['halted_at_bar']}"
    )


def demo_flexible_multi_leg_pairs_trading() -> None:
    print("\n=== 2. Flexible Multi-Leg Statistical Arbitrage (`flexible=True`, `row_wise=True`) ===")
    rng = np.random.default_rng(23)
    n_bars = 200
    common = np.cumsum(rng.normal(0.0004, 0.01, size=n_bars))
    spread = np.sin(np.linspace(0, 8 * np.pi, n_bars)) * 0.04 + rng.normal(0, 0.005, size=n_bars)
    asset_a = 100.0 * np.exp(common + 0.5 * spread)
    asset_b = 100.0 * np.exp(common - 0.5 * spread)
    close = np.column_stack([asset_a, asset_b])

    log_spread = np.log(asset_a / asset_b)
    spread_ma = alphax.indicators.ma(log_spread, 20).ma[:, 0]
    spread_std = alphax.indicators.mstd(log_spread, 20).mstd[:, 0]
    zscore = (log_spread - spread_ma) / np.where(spread_std > 0, spread_std, np.nan)

    # With `flexible=True`, `order_func(ctx, call, state)` is called for `call = 0, 1, ...` on each bar
    # until it returns `None`.
    def pairs_order_func(ctx: alphax.Context, call: int, state: None) -> tuple[int, dict] | None:
        _ = state
        if call >= 2:
            return None
        i = ctx.row
        z = zscore[i]
        if np.isnan(z):
            return None
        pos_a = ctx.position[0]
        notional_leg = 0.40 * ctx.value

        # Enter short spread (short A on call=0, long B on call=1) when z > +1.5
        if pos_a == 0.0 and z > 1.5:
            sign = -1.0 if call == 0 else +1.0
            return (call, {"size": sign * notional_leg, "size_type": "value", "direction": "both"})
        # Enter long spread (long A on call=0, short B on call=1) when z < -1.5
        if pos_a == 0.0 and z < -1.5:
            sign = +1.0 if call == 0 else -1.0
            return (call, {"size": sign * notional_leg, "size_type": "value", "direction": "both"})
        # Flatten both legs when spread mean-reverts inside |z| < 0.25
        if pos_a != 0.0 and abs(z) < 0.25:
            return (call, {"size": 0.0, "size_type": "targetamount", "direction": "both"})
        return None

    pf = alphax.from_order_func(
        close,
        pairs_order_func,
        flexible=True,
        row_wise=True,
        cash_sharing=True,
        init_cash=50_000.0,
        fees=0.0004,
    )
    print(
        f"Pairs Strategy Return: {pf.total_return()[0]:.2%} | Sharpe: {pf.sharpe_ratio()[0]:.2f} | "
        f"Total Fills: {pf.n_fills} | Max Net Exposure: {np.abs(pf.net_exposure()).max():.2%}"
    )


def demo_from_signals_callbacks() -> None:
    print("\n=== 3. `from_signals` Dynamic `signal_func` & Ratcheting `adjust_func` ===")
    n_bars = 120
    t = np.linspace(0, 4 * np.pi, n_bars)
    close = (100.0 + 12.0 * np.sin(t) + np.linspace(0, 15, n_bars)).reshape(-1, 1)
    high = close * 1.008
    low = close * 0.992

    def custom_signal_func(ctx: alphax.Context, k: int, state: None) -> tuple[bool, bool, bool, bool]:
        _ = state
        i = ctx.row
        pos = ctx.position[k]
        # Enter when price rises two bars in a row while flat; let adjust_func manage dynamic stop/target exits
        long_en = (pos == 0.0) and (i > 5) and bool(close[i, 0] > close[i - 1, 0] > close[i - 2, 0])
        return long_en, False, False, False

    def dynamic_ratchet_adjust(ctx: alphax.Context, k: int, stops: dict, state: None) -> dict | None:
        _ = state
        pos = ctx.position[k]
        if pos <= 0.0:
            return None
        entry_ref = stops["tp_price"]  # anchor price of the active position
        cur_price = ctx.val_price[k]
        if entry_ref > 0 and np.isfinite(entry_ref):
            unrealized_gain = (cur_price - entry_ref) / entry_ref
            # Tighten stop-loss from 4% to 1.5% trailing once unrealized profit exceeds +5%
            if unrealized_gain >= 0.05:
                return {"sl_stop": 0.015, "sl_trail": True, "tp_stop": 0.12}
            return {"sl_stop": 0.04, "sl_trail": False, "tp_stop": 0.12}
        return None

    pf = alphax.from_signals(
        close,
        high=high,
        low=low,
        sl_stop=0.04,
        tp_stop=0.12,
        signal_func=custom_signal_func,
        adjust_func=dynamic_ratchet_adjust,
        init_cash=10_000.0,
    )
    tr = pf.trades()
    win_rate = float((tr["pnl"] > 0).mean()) if len(tr["pnl"]) else 0.0
    print(
        f"Dynamic Signal+Adjust Return: {pf.total_return()[0]:.2%} | Trades: {pf.trade_count()[0]} | "
        f"Win Rate: {win_rate:.1%}"
    )


def main() -> None:
    demo_order_func_lifecycle_and_circuit_breaker()
    demo_flexible_multi_leg_pairs_trading()
    demo_from_signals_callbacks()


if __name__ == "__main__":
    main()
