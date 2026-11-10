"""Live Stepper, Incremental Resume, Chunked/Distributed Execution, Caching, Config & Portfolio Ops.

Scenario & Dimensions Covered:
  1. Live Bar-by-Bar Execution (`alphax.Stepper`):
     - `propose(close, size=..., **bar)` -> broker execution with partial fill / real-world
       slippage & fee -> `commit(fills)` -> exchange position `reconcile(broker_positions)`.
     - Exporting live `orders()`, `cash_records()`, `position`, and `cash`, then rebuilding a
       full `Portfolio` via `Portfolio.from_records`.
  2. Incremental Batch Continuation (`resume=pf` & `resume=SimState` -> `Segment`):
     - Extending a finished `Portfolio` onto longer history (`resume=pf_part1`), and
       advancing a low-level `SimState` (`resume=pf_part1.last_state, row_offset=100` -> `Segment`).
  3. Portfolio Algebra (`alphax.portfolios`) & Return Series Analytics (`alphax.returns`):
     - `Portfolio.from_records`, `select`, `column_stack`, `row_slice`, `resample_bars`,
       and `resample`.
     - Standalone `returns.metrics`, `returns.metric`, `returns.rolling`, `returns.value`,
       `returns.drawdowns`, and `returns.stats` directly on raw return arrays.
  4. Out-of-Core Chunked & Distributed Execution, Disk Caching & Persistence:
     - `chunked.run` with checkpointing and incremental history extension.
     - `distributed.run` across columns (`backend="thread"`).
     - `cache.Cache` (`memoize`, hit/miss `stats()`) and `alphax.save` / `alphax.load`.
  5. Declarative TOML Config (`alphax.config`), Cron Scheduler (`alphax.schedule`),
     and Introspection Tools (`alphax.check_shapes`, `alphax.help`, `alphax.docs`):
     - Running a complete backtest + parameter sweep from a declarative TOML file via `config.run`.
     - Defining recurring jobs (`schedule.Job`, `schedule.File`, `crontab`, `every`, `next_time`).
     - Pre-flight input validation with `check_shapes`, `alphax.help`, and offline doc search.

Run:
    uv run python examples/live_scale_and_ops.py
"""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import numpy as np
import alphax
from alphax import cache, chunked, config, distributed, docs, portfolios, returns, schedule


def demo_live_stepper() -> None:
    print("=== 1. Live Bar-by-Bar `alphax.Stepper` (Propose -> Broker Commit -> Reconcile) ===")
    stepper = alphax.Stepper(
        n_cols=2,
        init_cash=50_000.0,
        cash_sharing=True,
        fees=0.0005,
        account_type="margin",
        initial_margin=0.20,
        maintenance_margin=0.10,
    )

    prices = [
        np.array([100.0, 200.0]),
        np.array([103.0, 196.0]),
        np.array([106.0, 192.0]),
    ]
    orders_per_bar = [
        np.array([50.0, -25.0]),   # Bar 0: open long 50 col 0, short 25 col 1
        np.array([np.nan, np.nan]),  # Bar 1: hold and collect funding
        np.array([-47.5, 23.75]),  # Bar 2: flatten the 95%-filled positions
    ]
    for bar_idx, (close_bar, req_size) in enumerate(zip(prices, orders_per_bar)):
        proposal = stepper.propose(
            close_bar,
            size=req_size,
            funding_rate=np.array([0.0, 0.0002]),
        )
        if bar_idx == 0 and len(proposal["col"]) > 0:
            # Simulate broker partial fill (95% filled) and 2 bps adverse slippage on bar 0
            # Note: `propose` returns sparse fill records, while `commit(fills)` expects
            # per-column vectors `{"size": (n_cols,), "price": (n_cols,), "fees": (n_cols,)}`.
            filled_size = req_size * 0.95
            filled_price = close_bar * np.array([1.0002, 0.9998])
            filled_fees = np.abs(filled_size) * filled_price * 0.0005
            stepper.commit({"size": filled_size, "price": filled_price, "fees": filled_fees})
        else:
            stepper.commit()

        if bar_idx == 1:
            # Compare exchange-reported positions against Stepper's internal positions
            broker_pos = np.array([47.5, -23.75])
            drift = stepper.reconcile(broker_pos)
            print(f"  Bar {bar_idx} position reconciliation drift: {drift.tolist()} | Cash: ${stepper.cash[0]:,.2f}")

    print(
        f"Stepper completed {stepper.row} bars | Orders recorded: {len(stepper.orders()['idx'])} | "
        f"Cash records: {len(stepper.cash_records()['idx'])} | Final Positions: {stepper.position.tolist()} | "
        f"Final Cash: ${stepper.cash[0]:,.2f}"
    )


def demo_incremental_resume() -> None:
    print("\n=== 2. Incremental Batch Continuation (`resume=pf` & `resume=SimState`) ===")
    rng = np.random.default_rng(31)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=(200, 2)), axis=0))
    entries = np.zeros_like(close, dtype=bool)
    exits = np.zeros_like(close, dtype=bool)
    entries[10::35] = True
    exits[25::35] = True

    pf_full = alphax.from_signals(close, entries, exits, init_cash=10_000.0, fees=0.001)
    pf_part1 = alphax.from_signals(close[:100], entries[:100], exits[:100], init_cash=10_000.0, fees=0.001)

    # High-level continuation: pass the full extended history with `resume=pf_part1`
    pf_resumed = alphax.from_signals(close, entries, exits, resume=pf_part1, fees=0.001)

    # Low-level continuation: pass only the new slice `[99:]` (1 lookback bar) with `resume=pf_part1.last_state`
    seg_part2 = alphax.from_signals(
        close[99:],
        entries[99:],
        exits[99:],
        resume=pf_part1.last_state,
        row_offset=99,
        fees=0.001,
    )

    parity = np.allclose(pf_full.value(0), pf_resumed.value(0)) and np.allclose(pf_full.value(1), pf_resumed.value(1))
    print(
        f"Part 1 end positions: {list(pf_part1.last_state.position)} | "
        f"Part 2 Segment fills: {len(seg_part2.orders()['idx'])} | "
        f"Exact equity parity with full run: {parity}"
    )


def demo_portfolio_algebra_and_returns_module() -> None:
    print("\n=== 3. Portfolio Algebra (`alphax.portfolios`) & Standalone `alphax.returns` ===")
    rng = np.random.default_rng(44)
    n_bars = 96  # 4 days of 1-hour bars
    index = np.arange(
        np.datetime64("2025-03-01T00:00:00", "ns"),
        np.datetime64("2025-03-01T00:00:00", "ns") + np.timedelta64(n_bars, "h"),
        np.timedelta64(1, "h"),
    )
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, size=(n_bars, 3)), axis=0))
    fast = alphax.indicators.ma(close, 4).ma
    slow = alphax.indicators.ma(close, 12).ma
    pf = alphax.from_signals(
        close,
        alphax.crossed_above(fast, slow),
        alphax.crossed_above(slow, fast),
        index=index,
        init_cash=10_000.0,
    )

    pf_col0 = portfolios.select(pf, [0])
    pf_cols12 = portfolios.select(pf, [1, 2])
    pf_stacked = portfolios.column_stack(pf_col0, pf_cols12)
    pf_sliced = portfolios.row_slice(pf, 24, 72)
    bar_of_day = portfolios.resample_bars(index, "D")
    pf_daily = portfolios.resample(pf, bar_of_day)

    # Standalone `alphax.returns` analytics directly on raw return vectors without a Portfolio
    raw_rets = pf.returns(0)
    bench_rets = pf.returns(1)
    solo_names, paired_names = returns.metrics()
    sr_val = returns.metric(raw_rets, "sharpe_ratio", ann_factor=365 * 24)
    roll_sr = returns.rolling(raw_rets, "sharpe_ratio", window=24, ann_factor=365 * 24)
    comp_val = returns.value(raw_rets, start=10_000.0)
    dd_info = returns.drawdowns(raw_rets)
    ret_stats = returns.stats(raw_rets, benchmark=bench_rets, ann_factor=365 * 24)

    print(
        f"Stacked cols={pf_stacked.n_cols} | Sliced rows={pf_sliced.n_rows} | "
        f"Resampled 1D bars={pf_daily.n_rows}"
    )
    print(
        f"Standalone `alphax.returns` ({len(solo_names)} solo + {len(paired_names)} benchmark metrics): "
        f"Sharpe={sr_val:.2f}, Final Value=${comp_val[-1]:,.2f}, Rolling24h shape={roll_sr.shape}, "
        f"Drawdowns={len(dd_info['col'])}, Stats keys={len(ret_stats)}"
    )


def demo_chunked_distributed_cache_and_persistence() -> None:
    print("\n=== 4. Chunked & Distributed Execution, Disk Cache & Portfolio Persistence ===")
    rng = np.random.default_rng(77)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=(240, 4)), axis=0))
    fast = alphax.indicators.ma(close, 5).ma
    slow = alphax.indicators.ma(close, 15).ma
    entries = alphax.crossed_above(fast, slow)
    exits = alphax.crossed_above(slow, fast)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # 1. Chunked time-axis execution with checkpointing
        ckpt = tmp / "chunk.ckpt"
        pf_chunked = chunked.run(
            "from_signals",
            close,
            entries,
            exits,
            chunk=80,
            checkpoint=ckpt,
            init_cash=10_000.0,
        )
        # 2. Distributed column-axis execution across worker threads
        pf_dist = distributed.run(
            "from_signals",
            close,
            entries,
            exits,
            n_chunks=2,
            backend="thread",
            init_cash=10_000.0,
        )
        # 3. Disk-backed memoization cache for expensive indicator/factor computations
        disk_cache = cache.Cache(tmp / "ind_cache")

        @disk_cache.memoize
        def compute_heavy_rsi(arr: np.ndarray, window: int) -> np.ndarray:
            return alphax.indicators.rsi(arr, window=window).rsi

        _ = compute_heavy_rsi(close, 14)  # miss
        _ = compute_heavy_rsi(close, 14)  # hit

        # 4. Save and reload Portfolio state
        pf_file = tmp / "saved_portfolio.alphax"
        alphax.save(pf_dist, pf_file)
        pf_loaded = alphax.load(pf_file)

        parity = all(np.allclose(pf_chunked.value(c), pf_dist.value(c)) for c in range(4))
        print(
            f"Chunked vs Distributed Equity Parity: {parity} | "
            f"Loaded Portfolio Returns: {pf_loaded.total_return().round(4).tolist()}"
        )
        print(f"Disk Cache Stats: {disk_cache.stats()}")


def demo_declarative_config_schedule_and_tools() -> None:
    print("\n=== 5. Declarative TOML Config (`alphax.config`), Scheduler & Developer Tools ===")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # Write synthetic long-format CSV market data and a declarative TOML strategy config
        csv_path = tmp / "bars.csv"
        rng = np.random.default_rng(15)
        c = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=100)))
        lines = ["time,symbol,open,high,low,close,volume"]
        base_day = dt.date(2025, 1, 1)
        for i, v in enumerate(c):
            day = (base_day + dt.timedelta(days=i)).isoformat()
            lines.append(f"{day},BTC,{v * 0.999:.4f},{v * 1.005:.4f},{v * 0.995:.4f},{v:.4f},1000")
        csv_path.write_text("\n".join(lines))

        toml_path = tmp / "strategy.toml"
        toml_path.write_text(
            f"""
[data]
csv = "{csv_path.name}"

[signals]
entries = "Ref(Mean($close, {{fast}}), 1) > Ref(Mean($close, {{slow}}), 1)"
exits   = "Ref(Mean($close, {{fast}}), 1) < Ref(Mean($close, {{slow}}), 1)"

[backtest]
init_cash = 10000.0
fees = 0.0005
direction = "longonly"

[sweep]
fast = [5, 10]
slow = [20, 30]

[output]
stats = "stats.json"
portfolio = "last.alphax"
"""
        )
        cfg_res = config.run(toml_path)

        # Schedule recurring strategy jobs & export crontab
        log_path = tmp / "runs.log"
        job = schedule.Job(
            func=lambda: config.run(toml_path),
            name="daily_alpha_sweep",
            notify=[schedule.File(str(log_path))],
        )
        # Run scheduled job once via `schedule.every(..., times=1)` with zero sleep
        schedule.every(job, interval=0.0, times=1, sleep=lambda _: None)
        next_fire = schedule.next_time(dt.datetime(2025, 1, 1, 12, 15), at="16:30")
        cron_line = schedule.crontab(toml_path, at="16:30", weekdays="1-5")

        # Pre-flight shape validation, `alphax.help`, & offline BM25 doc search
        shape_report = alphax.check_shapes(
            c,
            entries=c > 100,
            force_exit=np.zeros(len(c), dtype=bool),
            size=np.array([1.0]),
        )
        help_snippet = alphax.help("indicators.rsi", echo=False).splitlines()[0]
        doc_hits = docs.search("deflated_sharpe", limit=2)

        last_pf = cfg_res["portfolios"][-1]
        print(
            f"Declarative TOML Sweep Combos: {len(cfg_res['params'])} | "
            f"Last Run Return: {last_pf.total_return()[0]:+.2%} | Trades: {int(last_pf.trade_count()[0])}"
        )
        print(
            f"Scheduled Job `{job.name}` logged ({log_path.stat().st_size} bytes) | "
            f"Next fire={next_fire.isoformat()}"
        )
        print(f"Crontab Entry: {cron_line}")
        print(
            f"check_shapes={shape_report} | help='{help_snippet}' | "
            f"Docs search hits={[h.name for h in doc_hits]}"
        )


def main() -> None:
    demo_live_stepper()
    demo_incremental_resume()
    demo_portfolio_algebra_and_returns_module()
    demo_chunked_distributed_cache_and_persistence()
    demo_declarative_config_schedule_and_tools()


if __name__ == "__main__":
    main()
