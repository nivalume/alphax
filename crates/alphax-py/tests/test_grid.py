"""alphax.FactorGrid: parameterized expressions over a grid, in one graph."""

from __future__ import annotations

import numpy as np
import alphax
import pytest


def panel(rows: int = 200, cols: int = 3, seed: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (rows, cols)), axis=0))


def test_each_case_matches_its_own_compile():
    close = panel()
    grid = alphax.FactorGrid.compile(
        {"z": "($close - Mean($close, {n})) / Std($close, {n})", "up": "$close > Mean($close, {n}) * {k}"},
        n=[5, 20],
        k=[1.0, 1.01],
    )
    assert grid.cases == [{"n": 5, "k": 1.0}, {"n": 5, "k": 1.01}, {"n": 20, "k": 1.0}, {"n": 20, "k": 1.01}]
    assert grid.lookbacks == {"z": [4, 4, 19, 19], "up": [4, 4, 19, 19]} and grid.lookback == 19
    wide = grid.evaluate({"close": close})
    n = close.shape[1]
    for k, case in enumerate(grid.cases):
        alone = alphax.FactorSet.compile({
            "z": f"($close - Mean($close, {case['n']})) / Std($close, {case['n']})",
            "up": f"$close > Mean($close, {case['n']}) * {case['k']!r}",
        }).evaluate({"close": close})
        for name in ("z", "up"):
            np.testing.assert_array_equal(wide[name][:, k * n:(k + 1) * n], alone[name], err_msg=f"{name} {case}")


def test_a_grid_feeds_one_shared_price_sweep():
    close = panel()
    grid = alphax.FactorGrid.compile({"fast": "Mean($close, {f})", "slow": "Mean($close, {s})"},
                                     alphax.params.grid(f=[5, 10], s=[20, 50], where=lambda p: p["f"] < p["s"]))
    grid = grid.with_warmup()
    v = grid.evaluate({"close": close})
    entries = alphax.crossed_above(v["fast"], v["slow"])
    exits = alphax.crossed_above(v["slow"], v["fast"])
    pf = alphax.from_signals(close, entries, exits, column_map=grid.column_map(close.shape[1]))
    assert pf.n_cols == len(grid.cases) * close.shape[1]
    # The second case alone, over the same prices.
    n = close.shape[1]
    alone = alphax.from_signals(close, entries[:, n:2 * n], exits[:, n:2 * n])
    for col in range(n):
        np.testing.assert_allclose(pf.value(n + col), alone.value(col))


def test_placeholders_can_name_fields_and_errors_name_the_case():
    close, volume = panel(), panel(seed=9)
    grid = alphax.FactorGrid.compile({"ma": "Mean(${field}, 3)"}, field=["close", "volume"])
    wide = grid.evaluate({"close": close, "volume": volume})["ma"]
    ma = alphax.FactorSet.compile({"c": "Mean($close, 3)", "v": "Mean($volume, 3)"}).evaluate(
        {"close": close, "volume": volume})
    np.testing.assert_array_equal(wide, np.hstack([ma["c"], ma["v"]]))
    with pytest.raises(ValueError, match=r"2\.5"):
        alphax.FactorGrid.compile({"ma": "Mean($close, {n})"}, n=[5, 2.5])
    with pytest.raises(ValueError, match="no value"):
        alphax.FactorGrid.compile({"ma": "Mean($close, {m})"}, n=[5])
    with pytest.raises(TypeError):
        alphax.FactorGrid.compile({"ma": "Mean($close, {n})"}, [{"n": 5}], n=[5])
