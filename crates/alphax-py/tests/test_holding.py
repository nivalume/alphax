"""Holding charges through the Python API: funding, financing, interest on what a
long borrowed, the borrow fee of a short, the mark price, and the helper that
turns annual rates into per-bar ones."""

import numpy as np
import pytest

import alphax

R = 1.0 / 1024.0
N = np.nan


def test_funding_is_paid_by_a_long_at_the_close_and_is_part_of_the_return():
    close = np.array([128.0, 160.0, 160.0, 144.0])
    pf = alphax.from_orders(
        close, size=np.array([[8.0], [np.nan], [np.nan], [-8.0]]), init_cash=1000.0,
        account_type="margin", initial_margin=0.5,
        funding_rate=np.array([R, R, np.nan, np.nan]),
    )
    records = pf.cash_records()
    np.testing.assert_array_equal(records["idx"], [0, 1])
    np.testing.assert_array_equal(records["amount"], [-1.0, -1.25])
    np.testing.assert_array_equal(records["kind"], [3, 3])
    np.testing.assert_array_equal(pf.cash_flows("funding")[:, 0], [-1.0, -1.25, 0.0, 0.0])
    assert pf.cash_flows("funding").shape == (4, 1)
    np.testing.assert_allclose(np.ravel(pf.value()), [999.0, 1253.75, 1253.75, 1125.75])
    np.testing.assert_allclose(np.ravel(pf.free_cash()), [487.0, 741.75, 741.75, 1125.75])


def test_a_short_receives_funding_on_its_value_at_the_mark_price():
    pf = alphax.from_signals(
        np.array([128.0, 128.0]), np.array([True, False]), np.array([False, False]),
        direction="shortonly", size=8.0, init_cash=1000.0,
        account_type="margin", initial_margin=0.5,
        funding_rate=np.array([R, R]), mark_price=np.array([256.0, 512.0]),
    )
    # Quiet bar 1 is charged too.
    np.testing.assert_array_equal(pf.cash_records()["amount"], [2.0, 4.0])


def test_financing_charges_each_side_its_own_rate():
    close = np.full((2, 2), 128.0)
    pf = alphax.from_orders(
        close, size=np.array([[8.0, -8.0], [np.nan, np.nan]]), init_cash=1000.0,
        account_type="margin", initial_margin=0.5,
        long_financing_rate=R, short_financing_rate=-R / 2,
    )
    flows = pf.cash_flows("financing")
    np.testing.assert_array_equal(flows[:, 0], [-1.0, -1.0])
    np.testing.assert_array_equal(flows[:, 1], [0.5, 0.5])


def test_a_leveraged_long_pays_interest_and_a_cash_short_its_borrow_fee():
    close = np.full((3, 2), 128.0)
    size = np.array([[12.0, -8.0], [np.nan, np.nan], [-12.0, 8.0]])
    pf = alphax.from_orders(
        close, size=size, init_cash=1024.0, leverage=2.0, leverage_mode="eager",
        debt_rate=np.array([R, R, R]), borrow_rate=R,
    )
    orders = pf.orders()
    # 12 at 128 is 1536, half of it borrowed.
    np.testing.assert_array_equal(orders["debt"][orders["col"] == 0], [768.0, 0.0])
    np.testing.assert_array_equal(pf.cash_flows("interest")[:, 0], [-0.75, -0.75, 0.0])
    np.testing.assert_array_equal(pf.cash_flows("borrowfee")[:, 1], [-1.0, -1.0, 0.0])
    assert not pf.cash_flows("interest")[:, 1].any()


def test_a_charge_makes_an_unlevered_run_keep_balances_and_fill_as_before():
    close = np.full(3, 128.0)
    size = np.array([[-8.0], [np.nan], [8.0]])
    plain = alphax.from_orders(close, size=size, init_cash=1000.0).orders()
    charged = alphax.from_orders(close, size=size, init_cash=1000.0, borrow_rate=R).orders()
    assert "debt" not in plain
    np.testing.assert_array_equal(charged["debt"], [1024.0, 0.0])
    for field in ["col", "idx", "size", "price", "fees", "side"]:
        np.testing.assert_array_equal(plain[field], charged[field])


def test_market_shaped_rates_go_through_the_column_map():
    # Two symbols, swept twice: logical column k reads source column k % 2.
    close = np.full((2, 2), 128.0)
    funding = np.array([[R, 2 * R], [np.nan, np.nan]])
    pf = alphax.from_orders(
        close, size=np.array([[8.0] * 4, [np.nan] * 4]), init_cash=1000.0,
        account_type="margin", initial_margin=0.5,
        funding_rate=funding, column_map=np.array([0, 1, 0, 1]),
    )
    np.testing.assert_array_equal(pf.cash_flows("funding")[0], [-1.0, -2.0, -1.0, -2.0])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(funding_rate=np.zeros((3, 1))), r"funding_rate must have the source close shape \(2, 1\)"),
        (dict(mark_price=np.ones((2, 2))), "mark_price must have the source close shape"),
        (dict(borrow_rate=np.inf), "borrow_rate is infinite"),
        (dict(debt_rate=np.ones(5)), "per-row array has length 5, expected 2"),
    ],
)
def test_bad_holding_arguments_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        alphax.from_orders(
            np.full(2, 10.0), size=np.array([[-1.0], [np.nan]]), leverage=2.0, **kwargs
        )


def test_an_unknown_cash_flow_kind_is_refused():
    pf = alphax.from_orders(np.array([10.0]), size=1.0)
    with pytest.raises(ValueError, match="kind must be"):
        pf.cash_flows("swap")
    assert not pf.cash_flows("deposit").any()


def test_per_bar_charges_each_bar_for_the_time_until_the_next():
    # Thursday, Friday, Monday, Tuesday.
    idx = np.array(["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"], dtype="datetime64[D]")
    np.testing.assert_allclose(
        alphax.rates.per_bar(0.365, idx), [0.001, 0.003, 0.001, 0.0]
    )
    np.testing.assert_allclose(
        alphax.rates.per_bar(0.36, idx, days_per_year=360.0), [0.001, 0.003, 0.001, 0.0]
    )
    hourly = np.arange("2026-01-01T00", "2026-01-01T03", dtype="datetime64[h]")
    np.testing.assert_allclose(alphax.rates.per_bar(np.array([0.24, 0.48, 0.72]) * 365, hourly), [0.01, 0.02, 0.0])
    # A row gives each column its own rate; the result is ready for a matrix.
    both = alphax.rates.per_bar(np.array([[0.365, 0.73]]), idx)
    assert both.shape == (4, 2) and both.flags.f_contiguous
    np.testing.assert_allclose(both[:, 1], [0.002, 0.006, 0.002, 0.0])


def test_interest_over_a_weekend_is_three_days():
    # Daily bars, Thursday to Tuesday: Friday's bar holds the long over the
    # weekend and pays three days of 36.5% a year on the 768 borrowed.
    idx = np.array(["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"], dtype="datetime64[D]")
    pf = alphax.from_orders(
        np.full(4, 128.0), size=np.array([[12.0], [N], [N], [-12.0]]), init_cash=1024.0,
        leverage=2.0, leverage_mode="eager", debt_rate=alphax.rates.per_bar(0.365, idx),
    )
    np.testing.assert_array_equal(pf.orders()["debt"], [768.0, 0.0])
    np.testing.assert_allclose(pf.cash_flows("interest")[:, 0], [-0.768, -2.304, -0.768, 0.0])


def test_per_bar_accepts_a_pandas_index():
    pd = pytest.importorskip("pandas")
    idx = pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC")
    np.testing.assert_allclose(alphax.rates.per_bar(0.365, idx), [0.001, 0.001, 0.0])


@pytest.mark.parametrize(
    "rate, index, match",
    [
        (0.1, np.array([2, 1], dtype="datetime64[D]"), "strictly increasing"),
        (0.1, np.array([1.0, 2.0]), "datetimes"),
        (np.ones(3), np.array([1, 2], dtype="datetime64[D]"), "one value per bar"),
        (np.ones((3, 2)), np.array([1, 2], dtype="datetime64[D]"), "one row per bar"),
    ],
)
def test_per_bar_refuses_what_it_cannot_read(rate, index, match):
    with pytest.raises((TypeError, ValueError), match=match):
        alphax.rates.per_bar(rate, index)
