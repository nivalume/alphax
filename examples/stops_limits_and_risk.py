"""Advanced Execution, Resting Limit Orders, Stop Ladders & Built-in Risk Rules.

Scenario & Dimensions Covered:
  1. Realistic Execution & Order Controls:
     - `price="nextopen"` / `"nextvalidopen"`, `from_ago`, `bid_price` & `ask_price`,
       `price_area_vio_mode="cap"`.
     - Lot rounding (`size_granularity`), `min_size`, `max_size`, `cash_limit`,
       `allow_partial`, `lock_cash`, stochastic `reject_prob` + `seed`.
     - Audit logs (`log=True`, `pf.logs()`, `pf.log_reasons()`, `pf.reason_counts()`).
  2. Resting Limit Orders & Breakout Stop-Buy Orders in `from_signals`:
     - `order_type="limit"`, `limit_delta`, `delta_format="percent"`, `limit_reverse=True`,
       `limit_order_price="limit"`, `limit_delay`, `limit_tif` / `limit_expiry`
       (both row counts and `datetime64` `index=` durations), `upon_adj_limit_conflict`,
       `upon_opp_limit_conflict`, `order_mode=True`.
  3. Full Stop Engine, Stop Ladders & Stop-Limit Exits:
     - `sl_stop`, `sl_trail`, `tp_stop`, `tsl_stop` + `tsl_th`, `td_stop` & `dt_stop`.
     - `stop_entry_price`, `stop_exit_price`, `upon_stop_exit="reverse"`, `upon_stop_update`.
     - Multi-tranche `stop_ladder="uniform"` & `"adaptweighted"` with 2-D ladder thresholds.
     - `stop_order_type="limit"` with `stop_limit_delta`.
     - Pre-simulation `alphax.signals.generate_ohlc_stop_exits`.
  4. Built-in Rust Risk Rules & Excursion Analytics:
     - Volatility/ATR-sized risk budgeting (`risk` + `risk_stop`), automatic `breakeven`
       ratchet, portfolio `max_positions` and `max_exposure` caps (`cash_sharing=True`).
     - MFE/MAE excursions (`pf.excursions`, `pf.edge_ratio`, `pf.stats(0, high=..., low=...)`)
       and trade grouping modes (`trades="exit"`, `"entry"`, `"positions"`).

Run:
    uv run python examples/stops_limits_and_risk.py
"""

from __future__ import annotations

import numpy as np
import alphax


def make_intraday_ohlc(n_bars: int = 360, n_cols: int = 4, seed: int = 19) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.014, size=(n_bars, n_cols))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0.0, 0.002, size=(n_bars, n_cols)))
    span = np.abs(rng.normal(0.012, 0.005, size=(n_bars, n_cols)))
    high = np.maximum(open_, close) * (1.0 + span)
    low = np.minimum(open_, close) * (1.0 - span)
    bid = close * 0.9992
    ask = close * 1.0008
    # Hourly timestamps across 15 days
    index = np.arange(
        np.datetime64("2025-01-01T00:00:00", "ns"),
        np.datetime64("2025-01-01T00:00:00", "ns") + np.timedelta64(n_bars, "h"),
        np.timedelta64(1, "h"),
    )
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "bid": bid,
        "ask": ask,
        "index": index,
    }


def demo_execution_realism_and_audit_logs(mkt: dict[str, np.ndarray]) -> None:
    print("=== 1. Realistic Execution Controls, Bid/Ask Spread & Audit Logs ===")
    close = mkt["close"][:, :1]
    entries = np.zeros_like(close, dtype=bool)
    exits = np.zeros_like(close, dtype=bool)
    entries[10::25, 0] = True
    exits[20::25, 0] = True

    pf = alphax.from_signals(
        close,
        entries,
        exits,
        open=mkt["open"][:, :1],
        high=mkt["high"][:, :1],
        low=mkt["low"][:, :1],
        bid_price=mkt["bid"][:, :1],
        ask_price=mkt["ask"][:, :1],
        price="nextopen",
        price_area_vio_mode="cap",
        size=15.7,
        size_granularity=1.0,  # round to whole shares
        min_size=2.0,
        max_size=12.0,
        cash_limit=2_500.0,
        allow_partial=True,
        reject_prob=0.10,
        seed=42,
        log=True,
        init_cash=10_000.0,
    )
    logs = pf.logs()
    reasons = pf.reason_counts(0)
    orders = pf.orders()
    print(f"Filled orders: {pf.n_fills} (all sizes rounded & capped <= 12.0: {orders['size'][:5].tolist()})")
    print(f"Total bar audit logs: {pf.n_logs} (rows={len(logs['idx'])}) | Reason breakdown: {reasons}")


def demo_resting_limit_orders(mkt: dict[str, np.ndarray]) -> None:
    print("\n=== 2. Resting Limit Pullback & Breakout Orders with Datetime TIF ===")
    close = mkt["close"][:, :2]
    open_, high, low = mkt["open"][:, :2], mkt["high"][:, :2], mkt["low"][:, :2]
    index = mkt["index"]

    fast = alphax.indicators.ma(close, 8).ma
    slow = alphax.indicators.ma(close, 24).ma
    entries = alphax.crossed_above(fast, slow)
    exits = alphax.crossed_above(slow, fast)

    # Pullback limit order: place a resting buy 0.6% below signal price, expire after 6 hours or at 18:00 daily
    pf_pullback = alphax.from_signals(
        close,
        entries,
        exits,
        open=open_,
        high=high,
        low=low,
        index=index,
        order_type="limit",
        limit_delta=0.006,
        delta_format="percent",
        limit_order_price="limit",
        limit_delay=False,
        limit_tif=np.timedelta64(6, "h"),
        limit_expiry=np.datetime64("2025-01-10T12:00:00", "ns"),  # hard timestamp deadline (pure NumPy)
        upon_adj_limit_conflict="keepexecute",
        upon_opp_limit_conflict="cancelexecute",
        init_cash=10_000.0,
    )

    # Breakout stop-buy order: `limit_reverse=True` places limit ABOVE signal price, filling on upward breakout
    pf_breakout = alphax.from_signals(
        close,
        entries,
        exits,
        open=open_,
        high=high,
        low=low,
        order_type="limit",
        limit_delta=0.005,
        delta_format="percent",
        limit_reverse=True,
        limit_tif=8,
        init_cash=10_000.0,
    )
    print(f"Pullback Limit (6h TIF) trades: {pf_pullback.trade_count().tolist()} | Return: {pf_pullback.total_return().round(4).tolist()}")
    print(f"Breakout Limit (reverse) trades:{pf_breakout.trade_count().tolist()} | Return: {pf_breakout.total_return().round(4).tolist()}")


def demo_stop_ladders_and_time_stops(mkt: dict[str, np.ndarray]) -> None:
    print("\n=== 3. Stop Engine: Trailing, Time Stops, Stop Ladders & Pre-Sim Stops ===")
    close = mkt["close"][:, :1]
    open_, high, low = mkt["open"][:, :1], mkt["high"][:, :1], mkt["low"][:, :1]
    index = mkt["index"]

    entries = np.zeros_like(close, dtype=bool)
    exits = np.zeros_like(close, dtype=bool)
    entries[15::40, 0] = True

    # 3-rung adaptive take-profit ladder (2%, 4%, 6%) + trailing stop + 18-hour time stop
    tp_rungs = np.array([[0.02], [0.04], [0.06]])
    pf_ladder = alphax.from_signals(
        close,
        entries,
        exits,
        open=open_,
        high=high,
        low=low,
        index=index,
        sl_stop=0.025,
        tsl_stop=0.015,
        tsl_th=0.02,  # activate tight 1.5% trail once profit reaches +2%
        tp_stop=tp_rungs,
        stop_ladder="adaptweighted",
        td_stop=np.timedelta64(18, "h"),
        stop_entry_price="fillprice",
        stop_exit_price="stoplimit",
        upon_adj_stop_conflict="keepignore",
        init_cash=10_000.0,
    )

    # Compare trade grouping modes (`trades="exit"` vs `"positions"`) on partial ladder exits
    pf_pos_grouped = alphax.from_signals(
        close,
        entries,
        exits,
        open=open_,
        high=high,
        low=low,
        tp_stop=tp_rungs,
        stop_ladder="uniform",
        td_stop=20,
        trades="positions",
        init_cash=10_000.0,
    )

    # Standalone pre-simulation stop exit generator (`entries` first, `close` second)
    ladder_out: dict[str, np.ndarray] = {}
    stop_ex, stop_px, stop_kind = alphax.signals.generate_ohlc_stop_exits(
        entries,
        close,
        high=high,
        low=low,
        open=open_,
        sl_stop=0.025,
        sl_trail=True,
        tp_stop=0.05,
        out=ladder_out,
    )
    print(
        f"Ladder Portfolio Fills: {pf_ladder.n_fills} | Exit-Trades: {pf_ladder.trade_count()[0]} "
        f"| Position-Grouped Trades: {pf_pos_grouped.trade_count()[0]}"
    )
    print(
        f"Pre-Sim generate_ohlc_stop_exits: {entries.sum()} entries -> {stop_ex.sum()} stop exits "
        f"(mean trigger px={np.nanmean(stop_px):.2f}, kinds={np.unique(stop_kind[stop_ex]).tolist()})"
    )


def demo_risk_rules_and_excursions(mkt: dict[str, np.ndarray]) -> None:
    print("\n=== 4. Rust Risk Rules (Risk-Per-Trade, Breakeven, Max Positions/Exposure) & MFE/MAE ===")
    close, open_, high, low = mkt["close"], mkt["open"], mkt["high"], mkt["low"]
    atr_pct = alphax.indicators.atr(high, low, close, window=14).atr / close
    atr_stop = np.nan_to_num(2.0 * atr_pct, nan=0.03)

    fast = alphax.indicators.ma(close, 10).ma
    slow = alphax.indicators.ma(close, 30).ma
    entries = alphax.crossed_above(fast, slow)
    exits = alphax.crossed_above(slow, fast)

    pf = alphax.from_signals(
        close,
        entries,
        exits,
        open=open_,
        high=high,
        low=low,
        # Dynamic 2*ATR stop-loss matrix + scalar risk-per-trade sizing & breakeven ratchet
        sl_stop=atr_stop,
        risk=0.015,
        risk_stop=0.03,
        breakeven=0.025,
        # Portfolio-wide guardrails across the 4 shared-cash assets
        cash_sharing=True,
        max_positions=2,
        max_exposure=0.70,
        init_cash=50_000.0,
        fees=0.0005,
        log=True,
    )

    exc = pf.excursions(high=high, low=low)
    edge = pf.edge_ratio(high=high, low=low)
    stats_grp0 = pf.stats(group=0, high=high, low=low)
    print(
        f"Risk-Managed Portfolio Return: {pf.total_return()[0]:.2%} | Max Gross Exposure: "
        f"{pf.gross_long_short().max():.2%} (cap=70%)"
    )
    print(
        f"Excursions ({len(exc['mfe_ret'])} trades): mean MFE={exc['mfe_ret'].mean():.2%}, "
        f"mean MAE={exc['mae_ret'].mean():.2%}, Edge Ratio={edge[0]:.2f}"
    )
    print(
        f"Group 0 Stats MFE/MAE keys: avg_mfe_long={stats_grp0['avg_mfe_long']:.2%}, "
        f"avg_mae_long={stats_grp0['avg_mae_long']:.2%}"
    )


def main() -> None:
    mkt = make_intraday_ohlc()
    demo_execution_realism_and_audit_logs(mkt)
    demo_resting_limit_orders(mkt)
    demo_stop_ladders_and_time_stops(mkt)
    demo_risk_rules_and_excursions(mkt)


if __name__ == "__main__":
    main()
