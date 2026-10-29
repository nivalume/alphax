"""Score factors before simulating them.

A synthetic universe of 15 symbols, three industry groups and a log market-cap
exposure. Three factors:

- ``momentum`` is planted in the next bar's return. It should score.
- ``noise`` is independent of prices. It should not.
- ``leak`` is the forward return itself. Its rank IC at that horizon is 1,
  which is the check that the scorer is reading the panel it was given.

Nothing here is a strategy, and nothing is downloaded. The placebo shuffles
forward returns; it does not shuffle the factor into a signal.

Run:
    uv run python examples/factor_evaluation.py
"""

from __future__ import annotations

import numpy as np
import phasor


BARS = 180
SYMBOLS = 15
HORIZONS = (1, 5, 10)
DRAWS = 200
SEED = 7


def cell(table, factor: str, horizon: int, column: str) -> float:
    """One summary cell, whether pandas is installed or not."""
    if hasattr(table, "loc"):
        return float(table.loc[(factor, horizon), column])
    return float(table[(factor, horizon)][column])


def universe(n_bars: int = BARS, n_symbols: int = SYMBOLS, seed: int = SEED) -> dict[str, np.ndarray]:
    """Prices, industry labels, log market cap, and the three factors.

    ``momentum[t]`` is known on bar ``t``. With ``lag=1`` the one-bar forward
    return on that bar is ``close[t + 2] / close[t + 1] - 1``, and that return
    is built to contain ``momentum[t]``.
    """
    rng = np.random.default_rng(seed)
    industry_id = np.repeat(np.arange(3, dtype=np.float64), n_symbols // 3)
    industry = np.broadcast_to(industry_id, (n_bars, n_symbols)).copy()
    mcap = np.broadcast_to(np.log(rng.uniform(1e8, 1e11, size=n_symbols)), (n_bars, n_symbols)).copy()

    momentum = rng.normal(size=(n_bars, n_symbols))
    centered = momentum - momentum.mean(axis=1, keepdims=True)
    raw = rng.normal(0.0, 0.008, size=(n_bars, n_symbols))
    raw[1:] += 0.015 * centered[:-1]
    simple = np.clip(raw, -0.4, 0.4)
    close = np.empty((n_bars, n_symbols), dtype=np.float64)
    close[0] = 100.0
    close[1:] = 100.0 * np.cumprod(1.0 + simple[:-1], axis=0)

    open_ = close * np.exp(rng.normal(0.0, 0.002, close.shape))
    spread = np.clip(np.abs(rng.normal(0.004, 0.001, close.shape)), 0.001, 0.05)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.lognormal(12.0, 0.4, close.shape)
    vwap = (high + low + close) / 3.0
    noise = rng.normal(size=(n_bars, n_symbols))
    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "vwap": vwap,
        "industry": industry,
        "mcap": mcap,
        "momentum": momentum,
        "noise": noise,
    }


def main() -> None:
    data = universe()
    close = data["close"]
    leak = phasor.factor_eval.forward_returns(close, horizons=(1,), lag=1)[1]
    factors = {"momentum": data["momentum"], "noise": data["noise"], "leak": leak}

    report = phasor.factor_eval.evaluate(
        factors,
        close,
        horizons=HORIZONS,
        groups=data["industry"],
        exposures={"size": data["mcap"]},
    )
    print("=== summary: IC, rank IC, Newey-West t, p-value, neutral rank IC ===")
    summary = report.summary()
    print(summary)

    print("\n=== decay: mean rank IC of momentum at each horizon ===")
    print(report.decay("momentum"))

    print("\n=== monotonicity and quantile returns of momentum at h=5 ===")
    print("monotonicity", report.monotonicity("momentum", h=5))
    print(report.quantile_returns("momentum", h=5))

    print("\n=== correlation of the three factors ===")
    print(phasor.factor_eval.correlation(factors))

    print("\n=== redundancy against Alpha158 ===")
    library = phasor.alpha158().evaluate(
        {name: data[name] for name in ("open", "high", "low", "close", "volume", "vwap")}
    )
    redundancy = phasor.factor_eval.max_correlation(factors, library)
    print(redundancy)

    forward = report.forwards[1]
    print("\n=== placebo: permute forward returns, 200 draws ===")
    momentum_placebo = phasor.factor_eval.placebo(factors["momentum"], forward, draws=DRAWS, seed=0)
    noise_placebo = phasor.factor_eval.placebo(factors["noise"], forward, draws=DRAWS, seed=0)
    for name, result in (("momentum", momentum_placebo), ("noise", noise_placebo)):
        print(
            f"{name}: observed {result['observed']:.4f}  p_value {result['p_value']:.4f}  "
            f"null {result['null_distribution'].shape[0]}"
        )

    print("\n=== quantile backtest of momentum ===")
    book = phasor.factor_eval.quantile_backtest(close, factors["momentum"], quantiles=5, rebalance=5, lag=1)
    print(book.stats())

    momentum_ic = cell(summary, "momentum", 1, "rank_ic")
    momentum_p = cell(summary, "momentum", 1, "p_nw")
    noise_p = cell(summary, "noise", 1, "p_nw")
    noise_t = cell(summary, "noise", 1, "t_nw")
    leak_ic = cell(summary, "leak", 1, "rank_ic")
    assert momentum_ic > 0.0, momentum_ic
    assert momentum_p < 0.05, momentum_p
    assert noise_p > 0.05 or abs(noise_t) < 2.0, (noise_p, noise_t)
    assert abs(leak_ic - 1.0) < 1e-8, leak_ic
    assert momentum_placebo["observed"] > 0.0
    assert momentum_placebo["p_value"] < 0.05
    assert cell(summary, "momentum", 1, "neutral_rank_ic") > 0.0
    print("\nok")


if __name__ == "__main__":
    main()
