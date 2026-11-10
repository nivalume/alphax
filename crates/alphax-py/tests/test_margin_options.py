"""Isolated charges, interest on cash below zero, cash maintenance margin,
tiered maintenance margin, partial liquidation, auto-deleveraging and
per-column margin bases through the Python API."""

import numpy as np
import pytest

import alphax

N = np.nan


def col(*values):
    return np.array(values, dtype=float).reshape(-1, 1)


def liquidations(pf):
    o = pf.orders()
    flag = o["liquidation"]
    return list(zip(o["idx"][flag].tolist(), o["size"][flag].tolist(), o["price"][flag].tolist()))


def test_an_isolated_column_pays_funding_out_of_its_margin():
    # Short 1 at 100 posting 10; funding of 1% a bar is received.
    pf = alphax.from_orders(
        col(100, 100, 100, 100), col(-1, N, N, N), init_cash=1000.0, account_type="margin",
        initial_margin=0.1, margin_mode="isolated", funding_rate=0.01,
    )
    np.testing.assert_allclose(pf.cash_flows("funding").ravel(), [1, 1, 1, 1])
    # Into the margin, not the free cash.
    np.testing.assert_allclose(pf.free_cash(), [990, 990, 990, 990])
    np.testing.assert_allclose(pf.value(), [1001, 1002, 1003, 1004])


def test_cash_below_zero_pays_interest():
    # Long 1 at 100 on 10% margin with 10 of cash; at 50 the account owes 40.
    pf = alphax.from_orders(
        col(100, 50, 50), col(1, N, N), init_cash=10.0, account_type="margin",
        initial_margin=0.1, debt_rate=0.01,
    )
    np.testing.assert_allclose(pf.cash_flows("interest").ravel(), [0, -0.4, -0.404])


def test_a_borrowing_cash_column_is_liquidated_below_its_maintenance_margin():
    # Bought 2 at 100 with 100 of own cash and 100 borrowed: equity 2p − 100,
    # 25% maintenance 0.5p: breached below 66.67, which the close of 60 is.
    pf = alphax.from_orders(col(100, 90, 60), col(2, N, N), init_cash=100.0, leverage=2.0,
                            cash_maintenance_margin=0.25)
    assert liquidations(pf) == [(2, 2.0, 60.0)]
    # Without it the column is collateral, never closed.
    pf = alphax.from_orders(col(100, 90, 60), col(2, N, N), init_cash=100.0, leverage=2.0)
    assert liquidations(pf) == []


BAR = dict(open=col(100, 128), high=col(100, 130), low=col(100, 64))


def margin_long(**kwargs):
    # Long 4 at 100 posting all 100 of cash: equity 4p − 300 along the bar
    # from 128 to 64.
    return alphax.from_orders(col(100, 70), col(4, N), init_cash=100.0, account_type="margin",
                              initial_margin=0.25, **BAR, **kwargs)


def test_a_tier_table_moves_the_liquidation_price():
    # A flat 25%: M = p, breached at 100.
    assert liquidations(margin_long(maintenance_margin=0.25)) == [(1, 4.0, 100.0)]
    # 25% to a notional of 300, 50% beyond (amount 75): M = 2p − 75 above
    # p = 75, breached at 112.5.
    pf = margin_long(maintenance_margin_tiers=[(300.0, 0.25), (np.inf, 0.5)])
    [(idx, size, price)] = liquidations(pf)
    assert (idx, size) == (1, 4.0)
    assert price == pytest.approx(112.5)


def test_a_partial_liquidation_closes_in_steps_along_the_bar():
    # Half at 100; with 2 left, E = 2p − 100 and M = 0.5p, breached again at
    # 66.67, half of that; with 1 left the low of 64 is safe.
    out = liquidations(margin_long(maintenance_margin=0.25, liquidation_step=0.5))
    assert [(i, s) for i, s, _ in out] == [(1, 2.0), (1, 1.0)]
    assert out[0][2] == pytest.approx(100.0)
    assert out[1][2] == pytest.approx(200 / 3)


def test_auto_deleveraging_closes_at_the_mark():
    pf = alphax.from_orders(
        col(100, 110, 120), col(1, N, N), init_cash=1000.0, account_type="margin",
        initial_margin=0.1, auto_deleverage=np.array([False, False, True]),
    )
    assert liquidations(pf) == [(2, 1.0, 120.0)]
    assert pf.orders()["fees"][-1] == 0.0


def test_margin_basis_per_column():
    close = np.full((1, 2), 100.0)
    size = np.ones((1, 2))
    pf = alphax.from_orders(close, size, init_cash=1000.0, account_type="margin",
                            initial_margin=np.array([30.0, 0.1]), margin_basis=["unit", "notional"])
    np.testing.assert_allclose(pf.orders()["locked_cash"], [30.0, 10.0])
