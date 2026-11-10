"""Dynamic Asset Allocation, Calendar Signals, Trading Sessions & Multi-Resolution.

Scenario & Dimensions Covered:
  1. Multi-Asset Portfolio Allocation (`alphax.allocation`) + Periodic DCA Deposits:
     - Comparing all 5 dynamic weighting schemes (`equal_weight`, `inverse_volatility`,
       `min_variance`, `max_sharpe`, `risk_parity`) on a 5-asset global macro universe.
     - Using `rebalance_mask`, `weights(...)`, and `from_weights(...)` with monthly
       dollar-cost averaging (`cash_deposits`).
  2. Calendar & Signal Processing Utilities (`alphax.signals`):
     - Turn-of-month / weekly calendar entries (`signals.calendar`).
     - Signal hygiene & timing: `clean`, `exit_after`, `exit_on`, `ranges`, `rank`,
       `nth_index`, `normalized_average_index`, `stats`.
  3. Intraday Trading Session Masks & End-of-Day Flattening (`alphax.sessions`):
     - Filtering signals to NY regular equity hours (`09:30`–`16:00` America/New_York,
       DST-aware) and flattening all open positions at session close via `force_exit`.
     - Overnight futures session mask (`18:00`–`17:00`).
  4. Multi-Resolution Alignment & Resampling (`alphax.resolution`):
     - Generating signals on coarse 1-hour bars, aligning (`resolution.align`) to
       fine 15-minute bars so intrabar stop-loss vs take-profit order is resolved
       accurately, and resampling (`resolution.resample`) equity/returns back to 1-hour.

Run:
    uv run python examples/allocation_and_intraday.py
"""

from __future__ import annotations

import numpy as np
import alphax
from alphax import allocation, resolution, sessions, signals


def demo_dynamic_asset_allocation() -> None:
    print("=== 1. Dynamic Asset Allocation (5 Weighting Schemes + Periodic DCA) ===")
    rng = np.random.default_rng(101)
    n_bars, n_assets = 504, 5  # ~2 years of daily bars across 5 asset classes
    drifts = np.array([0.0005, 0.0004, 0.0002, 0.0003, 0.0006])
    vols = np.array([0.014, 0.011, 0.005, 0.008, 0.020])
    mkt = rng.normal(0.0002, 0.006, size=(n_bars, 1))
    rets = mkt + rng.normal(drifts, vols, size=(n_bars, n_assets))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))

    # Monthly DCA contribution ($500 added every 21 trading days)
    dca_deposits = np.zeros(n_bars)
    dca_deposits[21::21] = 500.0

    schemes = [
        ("equal_weight", allocation.equal_weight),
        ("inverse_volatility", allocation.inverse_volatility),
        ("min_variance", allocation.min_variance),
        ("max_sharpe", allocation.max_sharpe),
        ("risk_parity", allocation.risk_parity),
    ]
    for name, weight_fn in schemes:
        w = allocation.weights(close, weight_fn, every=21, lookback=63)
        pf = allocation.from_weights(
            close,
            w,
            init_cash=50_000.0,
            cash_deposits=dca_deposits,
            fees=0.0005,
        )
        # Inspect the last rebalance row's weights
        rebal_rows = np.flatnonzero(~np.isnan(w[:, 0]))
        last_w = w[rebal_rows[-1]] if len(rebal_rows) else np.zeros(n_assets)
        print(
            f"  {name:<20} Return={pf.total_return()[0]:>7.2%} | Sharpe={pf.sharpe_ratio()[0]:>5.2f} "
            f"| MaxDD={pf.max_drawdown()[0]:>6.2%} | LastWeights={np.round(last_w, 2).tolist()}"
        )


def demo_calendar_and_signal_utilities() -> None:
    print("\n=== 2. Calendar Signals & Signal Processing Utilities (`alphax.signals`) ===")
    n_bars = 120
    # Daily timestamps starting 2025-01-01
    index = np.arange(
        np.datetime64("2025-01-01T00:00:00", "ns"),
        np.datetime64("2025-01-01T00:00:00", "ns") + np.timedelta64(n_bars, "D"),
        np.timedelta64(1, "D"),
    )
    close = 100.0 + np.cumsum(np.sin(np.arange(n_bars) * 0.2))

    # Enter on the first bar of each week (`freq="W", which="first"`), exit on the 4th bar (`n=3`) or after 3 bars
    mon_entries = signals.calendar(index, freq="W", which="first", n=0)
    thu_exits = signals.calendar(index, freq="W", which="first", n=3)
    timed_exits = signals.exit_after(mon_entries, 3)
    combined_exits = signals.exit_on(mon_entries, thu_exits | timed_exits)
    clean_en, clean_ex = signals.clean(mon_entries, combined_exits)

    rngs = signals.ranges(clean_en, kind="between")
    sig_ranks = signals.rank(mon_entries, reset_by=thu_exits)
    sig_stats = signals.stats(clean_en)
    second_signal_bar = signals.nth_index(clean_en, 1)
    avg_pos = signals.normalized_average_index(clean_en)

    print(
        f"Calendar Weekly: {mon_entries.sum()} raw -> {clean_en.sum()} clean entries | "
        f"Between-signal gaps: {len(rngs['start_idx'])}, mean gap={rngs['duration'].mean():.1f} bars"
    )
    print(
        f"Signal stats: count={sig_stats['count'][0]}, rate={sig_stats['rate'][0]:.3f}, "
        f"2nd index={second_signal_bar}, normalized center={avg_pos:.3f}, max streak={sig_ranks.max()}"
    )
    _ = (close, clean_ex)


def demo_intraday_sessions_and_force_exit() -> None:
    print("\n=== 3. Intraday Trading Sessions & End-of-Session `force_exit` ===")
    # 3 days of 15-minute bars in UTC covering NY Regular Trading Hours (14:30 - 21:00 UTC in winter)
    n_bars = 3 * 96
    index = np.arange(
        np.datetime64("2025-01-06T00:00:00", "ns"),
        np.datetime64("2025-01-06T00:00:00", "ns") + np.timedelta64(n_bars * 15, "m"),
        np.timedelta64(15, "m"),
    )
    rng = np.random.default_rng(55)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.003, size=(n_bars, 1)), axis=0))

    rth_mask = sessions.session_mask(index, start="09:30", end="16:00", tz="America/New_York")
    rth_end = sessions.session_end(index, start="09:30", end="16:00", tz="America/New_York")
    globex_mask = sessions.session_mask(index, start="18:00", end="17:00", tz="America/New_York")

    fast = alphax.indicators.ma(close, 4).ma
    slow = alphax.indicators.ma(close, 12).ma
    # Note: `crossed_above` returns 2-D `(n_bars, n_cols)` while `session_mask` returns 1-D `(n_bars,)`,
    # so use `rth_mask[:, None]` when masking 2-D signals.
    entries = alphax.crossed_above(fast, slow) & rth_mask[:, None]
    exits = alphax.crossed_above(slow, fast) & rth_mask[:, None]

    pf = alphax.from_signals(
        close,
        entries,
        exits,
        force_exit=rth_end,
        init_cash=25_000.0,
        log=True,
    )
    overnight_exposure = np.abs(pf.assets(0)[~rth_mask]).max()
    print(
        f"NY RTH active bars: {rth_mask.sum()}/{n_bars} | Session-end bars: {rth_end.sum()} "
        f"| Globex bars: {globex_mask.sum()}"
    )
    print(
        f"Intraday Trades: {pf.trade_count()[0]} | Max overnight position outside RTH: "
        f"{overnight_exposure:.1f} (strictly flat overnight!)"
    )


def demo_multi_resolution_alignment() -> None:
    print("\n=== 4. Multi-Resolution Signal Alignment (1h -> 15m) & Equity Resampling ===")
    n_coarse = 60
    n_fine = n_coarse * 4
    coarse_idx = np.arange(
        np.datetime64("2025-02-01T00:00:00", "ns"),
        np.datetime64("2025-02-01T00:00:00", "ns") + np.timedelta64(n_coarse, "h"),
        np.timedelta64(1, "h"),
    )
    fine_idx = np.arange(
        np.datetime64("2025-02-01T00:00:00", "ns"),
        np.datetime64("2025-02-01T00:00:00", "ns") + np.timedelta64(n_fine * 15, "m"),
        np.timedelta64(15, "m"),
    )

    rng = np.random.default_rng(88)
    fine_close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, size=n_fine)))
    fine_high = fine_close * (1.0 + np.abs(rng.normal(0.003, 0.001, size=n_fine)))
    fine_low = fine_close * (1.0 - np.abs(rng.normal(0.003, 0.001, size=n_fine)))

    # Resample fine 15m close to coarse 1h close (`how="last"`) to compute 1h signals
    coarse_close = resolution.resample(fine_close, fine_idx, coarse_idx, how="last")
    fast_1h = alphax.indicators.ma(coarse_close, 4).ma[:, 0]
    slow_1h = alphax.indicators.ma(coarse_close, 10).ma[:, 0]
    coarse_entries = alphax.crossed_above(fast_1h, slow_1h)
    coarse_exits = alphax.crossed_above(slow_1h, fast_1h)

    # Align 1h triggers onto 15m grid (`fill=False` fires once per hour; `fill=True` forward-fills regime)
    fine_entries = resolution.align(coarse_entries, coarse_idx, fine_idx, fill=False)
    fine_exits = resolution.align(coarse_exits, coarse_idx, fine_idx, fill=False)
    fine_sl = resolution.align(np.full(n_coarse, 0.012), coarse_idx, fine_idx, fill=True)
    # Note: in `from_signals`, 1-D `sl_stop` is per-column `(n_cols,)`, so a time-varying stop for 1 column is `(n_fine, 1)`
    fine_sl_2d = fine_sl.reshape(-1, 1)

    pf_fine = alphax.from_signals(
        fine_close,
        fine_entries,
        fine_exits,
        high=fine_high,
        low=fine_low,
        sl_stop=fine_sl_2d,
        tp_stop=0.025,
        init_cash=10_000.0,
    )

    # Resample 15m equity and compound returns back to 1h reporting grid
    hourly_equity = resolution.resample(pf_fine.value(0), fine_idx, coarse_idx, how="last")
    hourly_rets = resolution.resample(pf_fine.returns(0), fine_idx, coarse_idx, how="compound")
    print(
        f"Coarse 1h entries: {coarse_entries.sum()} -> Aligned 15m entries: {fine_entries.sum()} | "
        f"Fine-grid Trades: {pf_fine.trade_count()[0]}"
    )
    print(
        f"Hourly Resampled Final Equity: ${hourly_equity[-1]:,.2f} | "
        f"Compounded 1h Return: {(np.prod(1.0 + hourly_rets) - 1.0):.2%}"
    )


def main() -> None:
    demo_dynamic_asset_allocation()
    demo_calendar_and_signal_utilities()
    demo_intraday_sessions_and_force_exit()
    demo_multi_resolution_alignment()


if __name__ == "__main__":
    main()
