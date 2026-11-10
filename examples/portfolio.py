"""Run from the repository root: python examples/portfolio.py.

Two offline examples over synthetic BTC/ETH/SOL prices:
1. One shared account, rebalanced every 20 bars to 30% per symbol.
2. Two moving-average parameter sets, each with its own three-symbol account.
"""

import alphax
import numpy as np


def main():
    symbols = ("BTCUSDT.SIM", "ETHUSDT.SIM", "SOLUSDT.SIM")
    rng = np.random.default_rng(92723)
    close = np.asfortranarray(
        100 * np.cumprod(1 + rng.normal(0.0002, 0.015, (400, len(symbols))), axis=0)
    )
    # Targets apply at this bar's close. NaN means no order; zero means exit.
    targets = np.full_like(close, np.nan)
    targets[::20] = 0.30  # 90% invested, leaving room for costs.
    allocated = alphax.from_orders(
        close,
        targets,
        size_type="targetpercent",
        cash_sharing=True,
        init_cash=100_000.0,
        call_seq="auto",
        fees=0.001,
    )
    assert allocated.n_cols == 3 and allocated.n_groups == 1
    holdings = np.column_stack([allocated.assets(col=c) for c in range(len(symbols))])
    np.testing.assert_allclose(
        allocated.value(group=0),
        allocated.cash(group=0) + (holdings * close).sum(axis=1),
    )
    print("1. Equal-weight account: BTCUSDT.SIM / ETHUSDT.SIM / SOLUSDT.SIM share one initial 100,000")
    print("Metrics:", allocated.metrics())
    print(f"Ending equity: {allocated.value(group=0)[-1]:,.2f}")
    print(f"Ending cash: {allocated.cash(group=0)[-1]:,.2f}")
    print(
        "Ending quantities:",
        {symbol: float(holdings[-1, c]) for c, symbol in enumerate(symbols)},
    )

    pairs = ((10, 40), (20, 80))
    formulas = {
        f"trend_{i}": f"Mean($close, {fast}) / Mean($close, {slow}) - 1"
        for i, (fast, slow) in enumerate(pairs)
    }
    factors = alphax.FactorSet.compile(formulas).evaluate({"close": close})
    entries, exits = [], []
    for name in formulas:
        score = factors[name]
        entry = np.zeros_like(score, dtype=bool)
        exit_ = np.zeros_like(score, dtype=bool)
        entry[1:] = score[:-1] > 0  # Decision at t, fill at t+1 close.
        exit_[1:] = score[:-1] < 0
        entries.append(entry)
        exits.append(exit_)
    # Columns: A-BTC, A-ETH, A-SOL, B-BTC, B-ETH, B-SOL.
    # Each parameter set gets its own 100,000, shared only within its group.
    pf = alphax.from_signals(
        close,
        np.concatenate(entries, axis=1),
        np.concatenate(exits, axis=1),
        size=25_000.0,
        size_type="value",
        cash_sharing=True,
        group_lens=[len(symbols)] * len(pairs),
        init_cash=[100_000.0] * len(pairs),
        call_seq="auto",
        fees=0.001,
        threads=2,
        column_map=np.tile(np.arange(len(symbols)), len(pairs)),
    )
    assert pf.n_cols == 6 and pf.group_lens == (3, 3) and pf.n_groups == 2
    metrics = pf.metrics()
    scores = pf.traded_only(metrics["sharpe_ratio"])
    print("\n2. Signal parameter sweep: two independent shared accounts")
    for group, params in enumerate(pairs):
        print(
            f"group={group}, params={params}: return={metrics['total_return'][group]:.2%}, "
            f"Sharpe={scores[group]:.3f}, trades={metrics['trade_count'][group]}, "
            f"ending equity={pf.value(group=group)[-1]:,.2f}"
        )
    if np.isfinite(scores).any():
        best = int(np.nanargmax(scores))
        print("Best parameter group:", pairs[best])
        print("Account statistics:", pf.stats(group=best))
    print("Synthetic data demonstrate account mechanics, not investment performance.")


if __name__ == "__main__":
    main()
