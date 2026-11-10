"""A synthetic moving-average parameter sweep through the Python API."""

import alphax
import numpy as np


@alphax.strategy
def ma_cross(data, fast=5, slow=20):
    values = alphax.factors(
        data,
        fast=f"Mean($close, {fast})",
        slow=f"Mean($close, {slow})",
    )
    return (
        alphax.crossed_above(values["fast"], values["slow"]),
        alphax.crossed_above(values["slow"], values["fast"]),
    )


def main():
    symbols = ("BTCUSDT.SIM", "ETHUSDT.SIM", "SOLUSDT.SIM", "AVAXUSDT.SIM")
    pairs = ((5, 20), (5, 50), (10, 50), (10, 100), (20, 100), (20, 200), (20, 1900))
    rng = np.random.default_rng(20260920)
    close = 100.0 * np.cumprod(
        1.0 + rng.normal(0.0, 0.011, (2000, len(symbols))), axis=0
    )

    portfolio = ma_cross.sweep(
        {"close": close},
        params=[{"fast": fast, "slow": slow} for fast, slow in pairs],
        init_cash=10_000.0,
        fees=0.001,
        threads=4,
    )

    labels = [(fast, slow, symbol) for fast, slow in pairs for symbol in symbols]
    sharpe = portfolio.traded_only(portfolio.sharpe_ratio())
    total = portfolio.total_return()
    trades = portfolio.trade_count()
    best = int(np.nanargmax(sharpe))
    fast, slow, symbol = labels[best]

    print(f"{close.shape[0]} bars x {len(symbols)} symbols -> {len(labels)} columns")
    print(
        f"best: fast={fast}, slow={slow}, symbol={symbol}, "
        f"sharpe={sharpe[best]:.2f}, return={total[best]:.1%}, trades={trades[best]}"
    )
    print(
        "synthetic prices: these numbers demonstrate the pipeline, not a strategy result"
    )


if __name__ == "__main__":
    main()
