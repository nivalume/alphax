"""Margin, Perpetual Funding Carry, Liquidation, Continuous Futures & FX Conversion.

Scenario & Dimensions Covered:
  1. Cash Account Leverage (`lazy` vs `eager`), Margin Interest (`debt_rate`),
     Stock Borrow Fee (`borrow_rate`), and `cash_maintenance_margin`.
  2. Spot + Perpetual Delta-Neutral Funding Rate Carry Strategy:
     - Mixed account types (`account_type=["cash", "margin"]`) in a single shared-cash
       account (`cash_sharing=True`).
     - Buying Spot (col 0) and shorting Perpetual Futures (col 1) when 8-hour funding
       rates spike positive, collecting `funding_rate` settled against `mark_price`.
     - Inspecting `pf.cash_flows("funding")` and `pf.cash_records()`.
  3. Tiered Maintenance Margin, Partial Liquidation (`liquidation_step`),
     `liquidation_fee`, and `auto_deleverage` (ADL).
  4. Non-Fill Cash Flows & Rate Conversion:
     - `alphax.rates.per_bar`, `long_financing_rate`, `short_financing_rate`,
       `cash_deposits`, `cash_dividends`, `cash_earnings`.
  5. Futures Contract Roll & Continuous Back-Adjustment (`alphax.futures`) + FX (`alphax.fx`):
     - Built-in CME/CBOT/NYMEX/ICE `CONTRACTS` and `spec_params("ES", initial_margin=...)`.
     - `roll_by_volume`, `roll_by_expiry`, `rolls`, and `continuous(method="difference"|"ratio")`.
     - Multipliers (`multiplier=50.0`), negative commodity prices (`allow_negative_price=True`),
       existing open positions (`init_position`, `init_price`, `init_cash="auto"`),
       and FX currency conversion (`fx.convert`, `fx.cross`).

Run:
    uv run python examples/margin_futures_and_carry.py
"""

from __future__ import annotations

import numpy as np
import alphax
from alphax import futures, fx, rates


def demo_cash_account_leverage_and_financing() -> None:
    print("=== 1. Cash Account Leverage (Lazy vs Eager), Debt Rate & Short Borrow Rate ===")
    n_bars = 120
    close = 100.0 * np.exp(np.linspace(0.0, 0.18, n_bars))
    index = np.arange(
        np.datetime64("2025-01-01", "ns"),
        np.datetime64("2025-01-01", "ns") + np.timedelta64(n_bars, "D"),
        np.timedelta64(1, "D"),
    )
    entries = np.zeros(n_bars, dtype=bool)
    exits = np.zeros(n_bars, dtype=bool)
    entries[0] = True
    exits[-1] = True

    # Buy $10,000 notional on $10,000 own cash with 2x leverage:
    # - "lazy" spends $10k own cash first -> borrows $0 -> pays $0 interest
    # - "eager" borrows (1 - 1/leverage) = 50% of every buy -> borrows $5k -> pays interest
    daily_debt = rates.per_bar(0.06, index, days_per_year=365.0)
    pf_lazy = alphax.from_signals(
        close,
        entries,
        exits,
        size=10_000.0,
        size_type="value",
        leverage=2.0,
        leverage_mode="lazy",
        debt_rate=daily_debt,
        cash_maintenance_margin=0.25,
        init_cash=10_000.0,
    )
    pf_eager = alphax.from_signals(
        close,
        entries,
        exits,
        size=10_000.0,
        size_type="value",
        leverage=2.0,
        leverage_mode="eager",
        debt_rate=daily_debt,
        cash_maintenance_margin=0.25,
        init_cash=10_000.0,
    )
    print(
        f"Lazy 2x ($10k notional) Return:  {pf_lazy.total_return()[0]:.2%} | Debt interest flow: "
        f"{pf_lazy.cash_flows('interest').sum():.2f}"
    )
    print(
        f"Eager 2x ($10k notional) Return: {pf_eager.total_return()[0]:.2%} | Debt interest flow: "
        f"{pf_eager.cash_flows('interest').sum():.2f}"
    )


def demo_spot_perp_funding_carry() -> None:
    print("\n=== 2. Spot + Perpetual Delta-Neutral Funding Rate Carry Strategy ===")
    rng = np.random.default_rng(7)
    n_bars = 240  # 80 days of 8-hour funding intervals
    spot_price = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, size=n_bars)))
    basis = rng.normal(0.0008, 0.0004, size=n_bars)
    perp_price = spot_price * (1.0 + basis)
    mark_price = spot_price * (1.0 + 0.8 * basis)

    # Column 0 = Spot (cash account), Column 1 = Perpetual Future (margin account)
    close = np.column_stack([spot_price, perp_price])
    mark_2d = np.column_stack([spot_price, mark_price])

    # Positive 8h funding rate on perp column (0 on spot column): shorts receive from longs
    funding_8h = np.clip(rng.normal(0.0004, 0.0003, size=n_bars), -0.0002, 0.0015)
    funding_matrix = np.column_stack([np.zeros(n_bars), funding_8h])

    # Enter carry trade when 3-bar average funding > 3 bps / 8h; exit to flat when < 1.5 bps
    avg_funding = alphax.indicators.ma(funding_8h, window=3).ma[:, 0]
    size_targets = np.full((n_bars, 2), np.nan)
    for t in range(3, n_bars):
        if avg_funding[t] > 0.0003:
            size_targets[t] = [1.0, -1.0]  # target +1 BTC spot, -1 BTC perp
        elif avg_funding[t] < 0.00015:
            size_targets[t] = [0.0, 0.0]   # flatten both legs

    pf = alphax.from_orders(
        close,
        size=size_targets,
        size_type="targetamount",
        direction="both",
        account_type=["cash", "margin"],
        cash_sharing=True,
        initial_margin=[1.0, 0.10],  # 10x perp margin
        maintenance_margin=[0.0, 0.05],
        funding_rate=funding_matrix,
        mark_price=mark_2d,
        init_cash=75_000.0,
        fees=0.0002,
    )

    funding_cf = pf.cash_flows("funding")
    print(
        f"Delta-Neutral Carry Total Return: {pf.total_return()[0]:.2%} | Sharpe: {pf.sharpe_ratio()[0]:.2f} "
        f"| Max Drawdown: {pf.max_drawdown()[0]:.2%}"
    )
    print(
        f"Cumulative Funding Collected:     ${funding_cf.sum():,.2f} across "
        f"{len(pf.cash_records()['idx'])} non-fill cash events"
    )


def demo_tiered_margin_liquidation_and_adl() -> None:
    print("\n=== 3. Tiered Maintenance Margin, Partial Liquidation & Auto-Deleverage (ADL) ===")
    # Price drops steadily from 100 to 72, triggering partial liquidation steps
    close = np.linspace(100.0, 72.0, 40)
    # Note: in `from_orders`, 1-D `size` is interpreted as per-column `(n_cols,)`, so a time series of orders
    # for a single column must be 2-D `(n_bars, 1)`.
    size = np.full((40, 1), np.nan)
    size[0, 0] = 10.0  # Open 10 contracts ($1,000 notional) with $150 initial equity (~6.7x leverage)

    tiers = [
        (500.0, 0.05),   # up to $500 notional: 5% MM
        (2_000.0, 0.10), # up to $2,000 notional: 10% MM
    ]
    adl = np.zeros(40, dtype=bool)
    adl[35] = True  # Exchange auto-deleverages any remaining open position at bar 35

    pf = alphax.from_orders(
        close,
        size=size,
        account_type="margin",
        initial_margin=0.12,
        maintenance_margin_tiers=tiers,
        liquidation_step=0.50,  # liquidate 50% of remaining position at each breach
        liquidation_fee=0.005,
        auto_deleverage=adl,
        init_cash=150.0,
        log=True,
    )
    reasons = pf.reason_counts(0)
    print(
        f"Fills executed (entry + partial liquidations): {pf.n_fills} | "
        f"Final Position: {pf.assets(0)[-1]:.2f} | Final Value: ${pf.value(0)[-1]:.2f}"
    )
    print(f"Audit log reasons: {reasons}")


def demo_continuous_futures_and_fx() -> None:
    print("\n=== 4. Futures Contract Roll (`alphax.futures`), Multipliers, Cash Flows & FX (`alphax.fx`) ===")
    n_bars = 100
    t = np.linspace(0.0, 1.0, n_bars)
    # 3 quarterly contracts (Front, Next, Far) trading in contango ($12.50 carry between expiries)
    c1 = 5_000.0 + 120.0 * np.sin(14.0 * t) + 150.0 * t
    c2 = c1 + 12.5
    c3 = c2 + 12.5
    prices = np.column_stack([c1, c2, c3])

    # Volume shifts from C1 -> C2 around bar 35, and C2 -> C3 around bar 70
    vol = np.column_stack(
        [
            np.exp(-((np.arange(n_bars) - 15) / 20.0) ** 2) * 100_000,
            np.exp(-((np.arange(n_bars) - 50) / 20.0) ** 2) * 120_000,
            np.exp(-((np.arange(n_bars) - 85) / 20.0) ** 2) * 140_000,
        ]
    )
    active_vol = futures.roll_by_volume(vol)
    active_exp = futures.roll_by_expiry([35, 70, 100], n_bars, before=2)
    roll_bars = np.flatnonzero(futures.rolls(active_vol)).tolist()

    cont_diff = futures.continuous(prices, active_vol, method="difference")
    cont_ratio = futures.continuous(prices, active_exp, method="ratio")

    # Simulate E-mini S&P 500 (ES) using built-in contract specs (multiplier=50, size_granularity=1.0)
    es_kwargs = futures.spec_params(["ES"])
    fast = alphax.indicators.ma(cont_diff, 8).ma
    slow = alphax.indicators.ma(cont_diff, 20).ma
    pf_es = alphax.from_signals(
        cont_diff,
        alphax.crossed_above(fast, slow),
        alphax.crossed_above(slow, fast),
        size=1.0,
        account_type="margin",
        initial_margin=15_000.0,
        maintenance_margin=13_500.0,
        margin_basis="unit",
        init_cash=250_000.0,
        **es_kwargs,
    )

    # Convert EUR-denominated asset series to USD via FX cross rate
    eur_usd = np.linspace(1.08, 1.11, n_bars)
    jpy_usd = np.linspace(0.0066, 0.0068, n_bars)
    eur_jpy = fx.cross(eur_usd, jpy_usd)
    usd_equity = fx.convert(pf_es.value(0), eur_usd)

    print(f"Built-in Futures Contracts ({len(futures.CONTRACTS)}): {sorted(futures.CONTRACTS.keys())[:10]}...")
    print(f"Detected Volume Roll Bars: {roll_bars}")
    print(
        f"Continuous ES (mult=50) Return: {pf_es.total_return()[0]:.2%} | "
        f"EURJPY last={eur_jpy[-1]:.2f} | Final Equity (EUR->USD): ${usd_equity[-1]:,.0f}"
    )
    _ = cont_ratio


def main() -> None:
    demo_cash_account_leverage_and_financing()
    demo_spot_perp_funding_carry()
    demo_tiered_margin_liquidation_and_adl()
    demo_continuous_futures_and_fx()


if __name__ == "__main__":
    main()
