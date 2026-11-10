"""Exposure, excursion and edge-ratio methods on ``Portfolio``.

The arithmetic is the engine's and is tested there. These pin what the
binding adds: shapes, memory order, which axis is accounts and which is
symbols, and how ``high``/``low`` are read back in.
"""

import numpy as np
import pytest

import alphax


def run(**kwargs):
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0.0, 1.0, (60, 3)), axis=0)
    entries = np.zeros(close.shape, dtype=bool)
    exits = np.zeros(close.shape, dtype=bool)
    entries[[2, 30], :] = True
    exits[[20, 50], :] = True
    high = close + rng.uniform(0.0, 2.0, close.shape)
    low = close - rng.uniform(0.0, 2.0, close.shape)
    pf = alphax.from_signals(close, entries, exits, init_cash=1000.0, size=1.0, **kwargs)
    return pf, close, high, low


def test_exposures_are_whole_matrices_in_column_order():
    pf, close, _, _ = run(direction="both")
    for name in ("allocations", "long_exposure", "short_exposure", "net_exposure", "gross_long_short"):
        m = getattr(pf, name)()
        assert m.shape == close.shape, name
        assert m.flags.f_contiguous, name
    long, short = pf.long_exposure(), pf.short_exposure()
    np.testing.assert_allclose(pf.net_exposure(), long - short)
    np.testing.assert_allclose(pf.gross_long_short(), long + short)


def test_an_allocation_is_the_position_value_over_the_account():
    pf, close, _, _ = run()
    for col in range(close.shape[1]):
        expected = pf.assets(col) * close[:, col] / pf.value(col)
        np.testing.assert_allclose(pf.allocations()[:, col], expected)


def test_exposures_are_per_account_in_a_shared_run():
    pf, close, _, _ = run(cash_sharing=True, group_lens=[2, 1])
    assert pf.allocations().shape == close.shape
    assert pf.long_exposure().shape == (close.shape[0], 2)
    # Each account's long exposure is the sum of its columns' allocations.
    alloc = pf.allocations()
    np.testing.assert_allclose(pf.long_exposure()[:, 0], alloc[:, 0] + alloc[:, 1])
    np.testing.assert_allclose(pf.long_exposure()[:, 1], alloc[:, 2])


def test_position_coverage_is_the_share_of_bars_held():
    pf, close, _, _ = run()
    held = np.array([np.mean(pf.assets(c) != 0.0) for c in range(close.shape[1])])
    np.testing.assert_allclose(pf.position_coverage(), held)


def test_excursions_line_up_with_the_trades():
    pf, _, high, low = run()
    ex = pf.excursions(high=high, low=low)
    n = len(pf.trades()["col"])
    assert set(ex) == {"mae", "mfe", "mae_ret", "mfe_ret"}
    for values in ex.values():
        assert values.shape == (n,)
    assert np.all(ex["mae"] <= 0.0) and np.all(ex["mfe"] >= 0.0)


def test_closes_only_can_only_understate_an_excursion():
    pf, _, high, low = run()
    wide = pf.excursions(high=high, low=low)
    narrow = pf.excursions()
    assert np.all(narrow["mfe"] <= wide["mfe"])
    assert np.all(narrow["mae"] >= wide["mae"])


def test_a_c_ordered_range_reads_the_same_as_an_f_ordered_one():
    """The copy reads by index, so row-major input is not a transposed range."""
    pf, _, high, low = run()
    f = pf.excursions(high=np.asfortranarray(high), low=np.asfortranarray(low))
    c = pf.excursions(high=np.ascontiguousarray(high), low=np.ascontiguousarray(low))
    for key in f:
        np.testing.assert_array_equal(f[key], c[key])


def test_edge_ratio_is_one_value_per_account():
    pf, close, high, low = run()
    edge = pf.edge_ratio(high, low)
    assert edge.shape == (close.shape[1],)
    shared, _, _, _ = run(cash_sharing=True, group_lens=[2, 1])
    assert shared.edge_ratio(high, low).shape == (2,)


def test_a_single_column_takes_a_1d_range():
    close = np.array([10.0, 11.0, 12.0, 11.0])
    pf = alphax.from_signals(close, np.array([True, False, False, False]), np.array([False, False, True, False]))
    ex = pf.excursions(high=close + 1.0, low=close - 1.0)
    np.testing.assert_allclose(ex["mfe"], [3.0])
    np.testing.assert_allclose(ex["mae"], [-1.0])


def test_a_misshaped_range_is_refused():
    pf, close, high, low = run()
    with pytest.raises(ValueError, match="high"):
        pf.excursions(high=high[:-1], low=low)
    with pytest.raises(ValueError, match="low"):
        pf.excursions(high=high, low=low[:, 0])
