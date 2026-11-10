"""Technical Indicators, Custom IndicatorFactory, Live Stream & Pattern Matching Strategy.

Scenario & Dimensions Covered:
  1. Built-in Rust Technical Indicators with Parameter Grids:
     - All 8 indicators (`ma`, `mstd`, `bbands`, `rsi`, `stoch`, `macd`, `atr`, `obv`) + `true_range`.
     - Cartesian (`product=True`) vs zip (`product=False`) multi-parameter sweeps.
     - `IndicatorResult.select()` and `.params()` inspection.
  2. Custom Python `IndicatorFactory`:
     - Building a multi-input, multi-output Keltner Squeeze indicator that inherits
       automatic parameter broadcasting, column tiling, `.params()`, and `.select()`.
  3. Live Bar-by-Bar Streaming (`alphax.indicators.Stream`):
     - Stateful `O(1)` incremental updates (`"rsi"`, `"macd"`, `"atr"`, `"bbands"`)
       with bit-identical parity against vectorized batch indicators.
  4. Multi-Window Chart Pattern Search & Pattern-Triggered Strategy (`alphax.patterns`):
     - Searching historical price series for a W-bottom / double-bottom template via
       `patterns.search(..., rescale="minmax", overlap=False)`.
     - Turning completed pattern matches into look-ahead-free entry signals via
       `patterns.signals(...)` and backtesting with `from_signals`.

Run:
    uv run python examples/indicators_and_patterns.py
"""

from __future__ import annotations

import numpy as np
import alphax
from alphax import indicators, patterns


def make_ohlcv(n_bars: int = 500, n_cols: int = 2, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, size=(n_bars, n_cols))
    close = 100.0 * np.exp(np.cumsum(rets, axis=0))
    open_ = close * np.exp(rng.normal(0.0, 0.004, size=(n_bars, n_cols)))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.007, 0.003, size=(n_bars, n_cols))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.007, 0.003, size=(n_bars, n_cols))))
    volume = rng.uniform(1_000.0, 10_000.0, size=(n_bars, n_cols))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def demo_builtin_indicators(data: dict[str, np.ndarray]) -> None:
    print("=== 1. Built-in Rust Technical Indicators & Parameter Grids ===")
    close, high, low, volume = data["close"], data["high"], data["low"], data["volume"]

    # Sweeping multiple windows & ewm flags via Cartesian product
    ma_res = indicators.ma(close, window=[10, 20, 50], ewm=[False, True], product=True)
    mstd_res = indicators.mstd(close, window=[10, 20])
    bb_res = indicators.bbands(close, window=[20], alpha=[1.5, 2.0, 2.5], product=True)
    rsi_res = indicators.rsi(close, window=[7, 14, 21])
    stoch_res = indicators.stoch(high, low, close, k_window=[14], d_window=[3, 5], product=True)
    macd_res = indicators.macd(close, fast_window=[8, 12], slow_window=[21, 26], signal_window=9, product=False)
    atr_res = indicators.atr(high, low, close, window=[14, 20])
    tr_arr = indicators.true_range(high, low, close)
    obv_res = indicators.obv(close, volume)

    print(f"MA grid combos:   {len(ma_res.params())} -> shape={ma_res.ma.shape}")
    print(f"MSTD shape:       {mstd_res.mstd.shape}")
    print(f"BBANDS combos:    {len(bb_res.params())} -> upper shape={bb_res.upper.shape}")
    print(f"RSI combos:       {len(rsi_res.params())} -> select(window=14) shape={rsi_res.select(window=14).rsi.shape}")
    print(f"STOCH combos:     {len(stoch_res.params())} -> %K/%D shape={stoch_res.percent_k.shape}")
    macd_hist = macd_res.macd - macd_res.signal
    print(f"MACD zip combos:  {macd_res.params()} -> hist shape={macd_hist.shape}")
    print(f"ATR combos:       {atr_res.params()} | TrueRange shape={tr_arr.shape} | OBV shape={obv_res.obv.shape}")


def demo_custom_indicator_factory(data: dict[str, np.ndarray]) -> None:
    print("\n=== 2. Custom IndicatorFactory (Keltner Band Width & Position) ===")

    # IndicatorFactory passes 1-D arrays (one column at a time) into `func`
    def keltner_compute(
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        window: int = 20,
        mult: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mid = indicators.ma(close, window=window, ewm=True).ma[:, 0]
        atr_val = indicators.atr(high, low, close, window=window).atr[:, 0]
        upper = mid + mult * atr_val
        lower = mid - mult * atr_val
        pct_b = (close - lower) / np.where(upper > lower, upper - lower, np.nan)
        return upper, lower, pct_b

    keltner = indicators.IndicatorFactory(
        keltner_compute,
        inputs=["close", "high", "low"],
        params=["window", "mult"],
        outputs=["upper", "lower", "pct_b"],
        defaults={"window": 20, "mult": 2.0},
    )

    res = keltner.run(
        data["close"],
        data["high"],
        data["low"],
        window=[15, 25],
        mult=[1.5, 2.0],
        product=True,
    )
    chosen = res.select(window=25, mult=2.0)
    print(f"Custom Keltner params: {res.params()}")
    print(f"Selected (window=25, mult=2.0) pct_b shape={chosen.pct_b.shape}, last={chosen.pct_b[-1].round(4).tolist()}")


def demo_streaming_indicators(data: dict[str, np.ndarray]) -> None:
    print("\n=== 3. Live Bar-by-Bar Streaming vs Batch Parity ===")
    close_1d = data["close"][:, 0]
    high_1d = data["high"][:, 0]
    low_1d = data["low"][:, 0]

    rsi_stream = indicators.Stream("rsi", n_cols=1, window=14)
    macd_stream = indicators.Stream("macd", n_cols=1, fast_window=12, slow_window=26, signal_window=9)
    atr_stream = indicators.Stream("atr", n_cols=1, window=14)
    bb_stream = indicators.Stream("bbands", n_cols=1, window=20, alpha=2.0)

    streamed_rsi = np.array([rsi_stream.update(c)["rsi"][0] for c in close_1d])
    streamed_macd = np.array([macd_stream.update(c)["macd"][0] for c in close_1d])
    streamed_atr = np.array(
        [atr_stream.update(c, high=h, low=l)["atr"][0] for h, l, c in zip(high_1d, low_1d, close_1d)]
    )
    streamed_bb_mid = np.array([bb_stream.update(c)["middle"][0] for c in close_1d])

    batch_rsi = indicators.rsi(close_1d, window=14).rsi[:, 0]
    batch_macd = indicators.macd(close_1d, 12, 26, 9).macd[:, 0]
    batch_atr = indicators.atr(high_1d, low_1d, close_1d, window=14).atr[:, 0]
    batch_bb_mid = indicators.bbands(close_1d, window=20, alpha=2.0).middle[:, 0]

    print(f"RSI parity:    allclose={np.allclose(streamed_rsi, batch_rsi, equal_nan=True)}")
    print(f"MACD parity:   allclose={np.allclose(streamed_macd, batch_macd, equal_nan=True)}")
    print(f"ATR parity:    allclose={np.allclose(streamed_atr, batch_atr, equal_nan=True)}")
    print(f"BBANDS parity: allclose={np.allclose(streamed_bb_mid, batch_bb_mid, equal_nan=True)}")


def demo_pattern_matching_strategy(data: dict[str, np.ndarray]) -> None:
    print("\n=== 4. Chart Pattern Search & Pattern-Driven Strategy ===")
    close = data["close"].copy()

    # Double-bottom (W-pattern) template: drop -> bounce -> retest low -> breakout
    w_bottom_template = np.array([1.0, 0.3, 0.1, 0.55, 0.15, 0.75, 1.0])

    # Search across multiple window scales (14, 21, 28 bars) without overlapping matches
    matches = patterns.search(
        close,
        w_bottom_template,
        window=[14, 21, 28],
        min_similarity=0.80,
        rescale="minmax",
        overlap=False,
    )
    print(f"Found {len(matches['col'])} non-overlapping W-bottom matches across windows [14, 21, 28]")
    if len(matches["col"]) > 0:
        print(
            f"  Top match: col={matches['col'][0]}, bars=[{matches['start_idx'][0]}..{matches['end_idx'][0]}], "
            f"window={matches['window'][0]}, similarity={matches['similarity'][0]:.3f}"
        )

    # Convert pattern completions into look-ahead-free entry signals at match['end_idx']
    entries = patterns.signals(matches, close.shape)
    exits = alphax.signals.exit_after(entries, 15)

    pf = alphax.from_signals(
        close,
        entries,
        exits,
        high=data["high"],
        low=data["low"],
        init_cash=10_000.0,
        fees=0.001,
        sl_stop=0.04,
        tp_stop=0.08,
    )
    tr = pf.trades()
    win_rate = float((tr["pnl"] > 0).mean()) if len(tr["pnl"]) else 0.0
    print(
        f"Pattern Strategy Trades: {pf.trade_count().tolist()} | "
        f"Total Return: {pf.total_return().round(4).tolist()} | "
        f"Overall Win Rate: {win_rate:.1%}"
    )


def main() -> None:
    data = make_ohlcv()
    demo_builtin_indicators(data)
    demo_custom_indicator_factory(data)
    demo_streaming_indicators(data)
    demo_pattern_matching_strategy(data)


if __name__ == "__main__":
    main()
