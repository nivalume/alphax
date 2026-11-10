"""Deposits and earnings through the Python API.

The engine-side accounting is tested in ``crates/alphax-analysis/tests/cash_flows.rs``.
These tests cover what only this layer decides: which shapes each input
accepts, how a shared run's per-account deposits are mapped onto columns,
and what ``Portfolio.cash_records()`` returns.
"""

import alphax
import numpy as np
import pytest

DEPOSIT, EARNING = 0, 1


def test_a_deposit_is_not_a_return_and_a_withdrawal_is_capped():
    close = np.array([10.0, 10.0, 20.0, 20.0])
    pf = alphax.from_orders(
        close,
        # (T, 1): a 1-D size would mean one value per column.
        size=np.array([[5.0], [np.nan], [np.nan], [np.nan]]),
        cash_deposits=np.array([0.0, 100.0, 0.0, -200.0]),
        cash_earnings=np.array([0.0, 0.0, 1.0, 0.0]),
    )
    np.testing.assert_array_equal(pf.cash(0), [50.0, 150.0, 151.0, 0.0])
    np.testing.assert_array_equal(pf.value(0), [100.0, 200.0, 251.0, 100.0])
    np.testing.assert_allclose(pf.returns(0), [0.0, 0.0, 0.255, 0.0], atol=1e-15)

    records = pf.cash_records()
    np.testing.assert_array_equal(records["idx"], [1, 2, 3])
    np.testing.assert_array_equal(records["amount"], [100.0, 1.0, -151.0])
    np.testing.assert_array_equal(records["kind"], [DEPOSIT, EARNING, DEPOSIT])
    # The value fell from 251 to 100 by withdrawal, not by loss: no drawdown
    # episode at all, which the stats table reports as NaN.
    assert np.isnan(pf.stats(0)["Max Drawdown [%]"])
    assert pf.metrics()["max_drawdown"][0] == 0.0


def test_one_dimensional_inputs_are_one_value_per_bar():
    # 1-D means a schedule here, not one value per column: three columns, four
    # bars, and the four values land on the four bars of every column.
    close = np.full((4, 3), 10.0)
    pf = alphax.from_orders(close, size=np.nan, cash_earnings=np.array([1.0, 0.0, 0.0, 2.0]))
    for col in range(3):
        np.testing.assert_array_equal(pf.cash(col), [101.0, 101.0, 101.0, 103.0])
    assert len(pf.cash_records()["amount"]) == 6


def test_shared_deposits_take_one_column_per_account():
    close = np.full((3, 4), 10.0)
    deposits = np.asfortranarray([[0.0, 0.0], [10.0, 20.0], [0.0, -5.0]])
    pf = alphax.from_orders(
        close,
        size=np.nan,
        cash_sharing=True,
        group_lens=[2, 2],
        cash_deposits=deposits,
    )
    np.testing.assert_array_equal(pf.cash(group=0), [100.0, 110.0, 110.0])
    np.testing.assert_array_equal(pf.cash(group=1), [100.0, 120.0, 115.0])
    # Booked to each account's first column.
    np.testing.assert_array_equal(pf.cash_records()["col"], [0, 2, 2])
    np.testing.assert_array_equal(pf.returns(group=1), [0.0, 0.0, 0.0])


def test_a_shared_deposit_matrix_of_the_wrong_width_is_refused():
    close = np.full((3, 4), 10.0)
    with pytest.raises(ValueError, match="n_groups"):
        alphax.from_orders(
            close,
            size=np.nan,
            cash_sharing=True,
            group_lens=[2, 2],
            cash_deposits=np.zeros((3, 4), order="F"),
        )


def test_signal_runs_earn_on_quiet_bars_and_match_across_threads():
    rng = np.random.default_rng(11)
    close = np.asfortranarray(10.0 + rng.random((30, 4)).cumsum(axis=0))
    entries = rng.random((30, 4)) > 0.8
    exits = rng.random((30, 4)) > 0.8
    earnings = np.asfortranarray(rng.normal(0.0, 1.0, (30, 4)))
    kwargs = dict(
        size=1.0,
        direction="both",
        cash_earnings=earnings,
        cash_deposits=np.where(np.arange(30) % 10 == 0, 50.0, 0.0),
    )
    serial = alphax.from_signals(close, entries, exits, **kwargs)
    threaded = alphax.from_signals(close, entries, exits, threads=3, **kwargs)
    for key, values in serial.cash_records().items():
        np.testing.assert_array_equal(values, threaded.cash_records()[key])
    for col in range(4):
        np.testing.assert_array_equal(serial.value(col), threaded.value(col))
    # Every non-zero earning on every bar was booked, signal or not, except
    # where a charge found no free cash at all.
    booked = serial.cash_records()
    assert (booked["kind"] == EARNING).sum() <= np.count_nonzero(earnings)
    assert (booked["kind"] == EARNING).sum() > 0.9 * np.count_nonzero(earnings)


def test_an_infinite_amount_is_refused():
    with pytest.raises(ValueError, match="infinite"):
        alphax.from_orders(np.full(3, 10.0), cash_earnings=np.inf)


def test_dividends_are_market_data_shared_through_column_map():
    # Two symbols, two parameter sets: dividends stay (T, 2) like close.
    close = np.full((3, 2), 10.0, order="F")
    dividends = np.asfortranarray([[0.0, 0.0], [1.0, 0.5], [0.0, 0.0]])
    # Buy on bar 0 only: 2 and 4 units in set 0, 1 and 1 in set 1.
    size = np.zeros((3, 4), order="F")
    size[0] = [2.0, 4.0, 1.0, 1.0]
    pf = alphax.from_orders(close, size=size, column_map=[0, 1, 0, 1], cash_dividends=dividends)
    records = pf.cash_records()
    np.testing.assert_array_equal(records["col"], [0, 1, 2, 3])
    np.testing.assert_array_equal(records["amount"], [2.0, 2.0, 1.0, 0.5])
    assert set(records["kind"]) == {2}
