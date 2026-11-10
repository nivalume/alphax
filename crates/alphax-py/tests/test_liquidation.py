"""Maintenance margin, liquidation and isolated margin through the Python API."""

import numpy as np
import pytest

import alphax

N = np.nan


def col(*values):
    return np.array(values, dtype=float).reshape(-1, 1)


def perpetual(**kwargs):
    """100 long at 100 on 10% margin out of 1000, a sixteenth maintenance."""
    base = dict(
        size=col(100.0, N, 1.0), init_cash=1000.0, account_type="margin",
        initial_margin=0.1, maintenance_margin=1 / 16, log=True,
    )
    base.update(kwargs)
    close = base.pop("close", col(100.0, 95.0, 95.0))
    return alphax.from_orders(close, **base)


def test_a_long_is_liquidated_where_its_path_crosses_the_maintenance_margin():
    # The equity 100p - 9000 meets 6.25p at 96, between the open and the low.
    pf = perpetual(
        open=col(100.0, 100.0, 95.0), high=col(100.0, 100.0, 96.0),
        low=col(100.0, 90.0, 94.0), liquidation_fee=0.0125,
    )
    orders = pf.orders()
    np.testing.assert_array_equal(orders["price"], [100.0, 96.0, 95.0])
    np.testing.assert_array_equal(orders["fees"], [0.0, 120.0, 0.0])
    np.testing.assert_array_equal(orders["liquidation"], [False, True, False])
    np.testing.assert_array_equal(orders["isolated"], [False, False, False])
    logs = pf.logs()
    np.testing.assert_array_equal(logs["liquidation"], [False, True, False, False])
    np.testing.assert_allclose(np.ravel(pf.value()), [1000.0, 480.0, 480.0])


def test_a_gap_through_the_bankruptcy_price_is_written_off():
    pf = perpetual(open=col(100.0, 80.0, 80.0), close=col(100.0, 80.0, 80.0))
    np.testing.assert_array_equal(pf.orders()["price"], [100.0, 80.0])
    np.testing.assert_array_equal(pf.cash_flows("liquidation")[:, 0], [0.0, 1000.0, 0.0])
    np.testing.assert_array_equal(pf.cash_records()["kind"], [7])
    np.testing.assert_allclose(np.ravel(pf.value()), [1000.0, 0.0, 0.0])


def test_without_ohlc_the_close_is_tested():
    pf = perpetual(close=col(100.0, 94.0, 94.0))
    np.testing.assert_array_equal(pf.orders()["liquidation"], [False, True, False])
    np.testing.assert_array_equal(pf.orders()["price"], [100.0, 94.0, 94.0])


def test_a_mark_price_is_what_is_tested():
    kwargs = dict(open=col(100.0, 100.0, 95.0), low=col(100.0, 90.0, 94.0))
    assert perpetual(**kwargs).orders()["liquidation"].any()
    calm = perpetual(**kwargs, mark_price=col(100.0, 99.0, 99.0))
    assert not calm.orders()["liquidation"].any()


def test_an_isolated_columns_profit_is_not_free_cash():
    # Without a maintenance margin, so neither is liquidated.
    close = col(100.0, 95.0, 105.0)
    kwargs = dict(close=close, size=col(100.0, N, N), init_cash=2000.0, maintenance_margin=None)
    iso = perpetual(**kwargs, margin_mode="isolated")
    np.testing.assert_allclose(np.ravel(iso.free_cash()), [1000.0, 1000.0, 1000.0])
    np.testing.assert_array_equal(iso.orders()["isolated"], [True])
    cross = perpetual(**kwargs, margin_mode="cross")
    np.testing.assert_allclose(np.ravel(cross.free_cash()), [1000.0, 500.0, 1500.0])
    np.testing.assert_allclose(np.ravel(iso.value()), np.ravel(cross.value()))


def test_an_isolated_column_is_liquidated_alone_and_loses_at_most_its_margin():
    close = np.array([[100.0, 100.0], [85.0, 99.0]])
    pf = alphax.from_orders(
        close, size=np.array([[100.0, 100.0], [N, N]]), open=close, init_cash=3000.0,
        account_type="margin", initial_margin=0.1, maintenance_margin=1 / 16,
        margin_mode=["isolated", "cross"], cash_sharing=True,
    )
    orders = pf.orders()
    np.testing.assert_array_equal(orders["col"], [0, 1, 0])
    np.testing.assert_array_equal(orders["liquidation"], [False, False, True])
    # 1500 lost on 1000 posted: 500 is written off to the isolated column.
    np.testing.assert_array_equal(pf.cash_flows("liquidation")[1], [500.0, 0.0])


def test_a_stop_inside_the_liquidation_price_closes_the_position_first():
    def run(sl_stop):
        return alphax.from_signals(
            np.array([100.0, 99.0]), np.array([True, False]), np.array([False, False]),
            open=np.array([100.0, 100.0]), high=np.array([100.0, 100.0]),
            low=np.array([100.0, 90.0]), size=100.0, init_cash=1000.0, sl_stop=sl_stop,
            account_type="margin", initial_margin=0.1, maintenance_margin=1 / 16,
        ).orders()

    np.testing.assert_array_equal(run(0.03)["price"], [100.0, 97.0])
    np.testing.assert_array_equal(run(0.03)["liquidation"], [False, False])
    np.testing.assert_array_equal(run(0.08)["price"], [100.0, 96.0])
    np.testing.assert_array_equal(run(0.08)["liquidation"], [False, True])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(margin_mode="hedged"), "margin_mode must be 'cross' or 'isolated'"),
        (dict(margin_mode=["cross"] * 2), "margin_mode has 2 entries for 1 columns"),
        (dict(margin_mode=[1]), "margin_mode must be"),
        (dict(maintenance_margin=1.0), "maintenance_margin must be"),
        (dict(maintenance_margin=-0.1), "maintenance_margin must be"),
        (dict(liquidation_fee=-0.1, close=col(100.0, 50.0, 50.0)), "liquidation_fee must be"),
        (dict(margin_mode=np.array(["cross", "isolated"])[None, :]), "margin_mode"),
    ],
)
def test_bad_liquidation_arguments_are_refused(kwargs, match):
    with pytest.raises((TypeError, ValueError), match=match):
        perpetual(**kwargs)


def test_a_run_without_a_maintenance_margin_is_unchanged():
    plain = alphax.from_orders(
        col(100.0, 50.0, 50.0), size=col(100.0, N, 1.0), init_cash=1000.0,
        account_type="margin", initial_margin=0.1,
    ).orders()
    again = perpetual(close=col(100.0, 50.0, 50.0), maintenance_margin=None, log=None).orders()
    for field in ["col", "idx", "size", "price", "fees", "side", "debt", "locked_cash"]:
        np.testing.assert_array_equal(plain[field], again[field])
    assert not again["liquidation"].any()
