"""alphax.Stepper: one bar at a time, booking reported fills."""

import numpy as np
import pytest

import alphax


def _market(n=50, cols=3, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, cols)), axis=0))
    size = np.where(rng.random((n, cols)) < 0.4, rng.normal(0, 1, (n, cols)), np.nan)
    return close, size


@pytest.mark.parametrize(
    "params",
    [
        dict(init_cash=1_000.0, fees=0.001, price=-np.inf),
        dict(init_cash=1_000.0, leverage=3.0, debt_rate=0.001, group_lens=[2, 1], cash_sharing=True),
        dict(init_cash=500.0, reject_prob=0.3, seed=9, cash_earnings=0.2),
    ],
)
def test_committing_every_proposal_is_one_run(params):
    close, size = _market()
    high, low = close * 1.01, close * 0.99
    one = alphax.from_orders(
        np.asfortranarray(close), size, high=np.asfortranarray(high), low=np.asfortranarray(low),
        **params,
    )
    st = alphax.Stepper(close.shape[1], **params)
    for i in range(len(close)):
        st.propose(close[i], size=size[i], high=high[i], low=low[i])
        st.commit()
    a, b = one.orders(), st.orders()
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    ca, cb = one.cash_records(), st.cash_records()
    for k in ca:
        np.testing.assert_array_equal(np.sort(ca[k]), np.sort(cb[k]), err_msg=k)
    np.testing.assert_array_equal(st.position, one.last_state.position)
    assert st.state.to_bytes() == one.last_state.to_bytes()


def test_reported_fills_are_what_is_booked():
    st = alphax.Stepper(2, init_cash=1_000.0, fees=0.01)
    proposal = st.propose([100.0, 50.0], size=[2.0, -1.0])
    np.testing.assert_allclose(proposal["size"], [2.0, 1.0])
    # The broker filled 1.5 at 100.2 with a fee of 0.3, and nothing else.
    booked = st.commit({"size": [1.5, np.nan], "price": [100.2, np.nan], "fees": [0.3, 0.0]})
    np.testing.assert_allclose(booked["size"], [1.5])
    np.testing.assert_allclose(booked["price"], [100.2])
    np.testing.assert_allclose(booked["fees"], [0.3])
    np.testing.assert_allclose(st.position, [1.5, 0.0])
    np.testing.assert_allclose(st.cash, [1_000.0 - 1.5 * 100.2 - 0.3, 1_000.0])
    np.testing.assert_allclose(st.reconcile([1.5, 0.0]), [0.0, 0.0])
    np.testing.assert_allclose(st.reconcile([1.0, 2.0]), [-0.5, 2.0])
    # The next bar continues from what was booked, not from the proposal.
    st.propose([101.0, 51.0], size=[-1.5, np.nan])
    st.commit()
    np.testing.assert_allclose(st.position, [0.0, 0.0])
    o = st.orders()
    np.testing.assert_array_equal(o["idx"], [0, 1])


def test_what_the_stepper_refuses():
    st = alphax.Stepper(1, init_cash=100.0)
    with pytest.raises(RuntimeError, match="propose"):
        st.commit()
    st.propose(10.0, size=1.0)
    with pytest.raises(ValueError, match="rejected"):
        st.commit({"size": [50.0], "price": [10.0]})
    with pytest.raises(ValueError, match="1 values"):
        st.propose([1.0, 2.0])
    with pytest.raises(TypeError, match="per bar"):
        alphax.Stepper(1, size=1.0)
