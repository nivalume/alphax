"""Function strategies should equal the explicit signal-engine path."""

import inspect

import alphax
import numpy as np
import pytest


@alphax.strategy
def threshold(data, level=100):
    close = data["close"]
    return close > level, close < level


def test_direct_call_and_factor_fields():
    close = np.array([[90.0, 110.0], [105.0, 95.0], [115.0, 85.0]])
    assert inspect.signature(threshold).parameters["level"].default == 100
    entries, exits = threshold({"close": close})
    np.testing.assert_array_equal(entries, close > 100)
    np.testing.assert_array_equal(exits, close < 100)
    np.testing.assert_allclose(alphax.factors({"close": close}, twice="$close * 2")["twice"], close * 2)
    # Warm-up: no value from a partial window.
    ma = alphax.factors({"close": close}, ma="Mean($close, 3)")["ma"]
    assert np.isnan(ma[:2]).all() and not np.isnan(ma[2:]).any()
    with pytest.raises(ValueError, match="future"):
        alphax.factors({"close": close}, future="Ref($close, -1)")
    with pytest.raises(ValueError, match="missing field"):
        alphax.factors({"close": close}, volume="$volume")


def test_crossed_above_engine_boundaries():
    left = np.array([2.0, 0.0, 1.0, 2.0, np.nan, 2.0, 0.0, 2.0])
    result = alphax.crossed_above(left, np.ones_like(left))
    assert result.shape == (len(left), 1)
    np.testing.assert_array_equal(result[:, 0], [False, False, False, True, False, False, False, True])
    with pytest.raises(ValueError, match="same"):
        alphax.crossed_above(left, left[:-1])
    assert alphax.crossed_above(np.array([]), np.array([])).shape == (0, 1)


@pytest.mark.parametrize("delay", [0, 1, 20])
@pytest.mark.parametrize("sharing", [False, True])
def test_sweep_matches_explicit_engine(delay, sharing):
    close = np.array([
        [90.0, 110.0], [105.0, 95.0], [115.0, 85.0],
        [95.0, 105.0], [110.0, 90.0], [80.0, 120.0],
    ])
    data = {"close": close}
    params = [{"level": 100}, {"level": 105}]
    original = threshold(data)
    pf = threshold.sweep(data, params=params, signal_delay=delay, init_cash=1000,
                         size=1, cash_sharing=sharing)
    entries, exits = [], []
    for group in params:
        raw_entry, raw_exit = threshold(data, **group)
        entry = np.zeros_like(raw_entry)
        exit_ = np.zeros_like(raw_exit)
        if delay == 0:
            entry[:], exit_[:] = raw_entry, raw_exit
        elif delay < len(close):
            entry[delay:], exit_[delay:] = raw_entry[:-delay], raw_exit[:-delay]
        entries.append(entry)
        exits.append(exit_)
    explicit = alphax.from_signals(close, np.concatenate(entries, axis=1),
                                 np.concatenate(exits, axis=1),
                                 column_map=[0, 1, 0, 1], init_cash=1000, size=1,
                                 cash_sharing=sharing,
                                 group_lens=[2, 2] if sharing else None)
    for field in pf.orders():
        np.testing.assert_array_equal(pf.orders()[field], explicit.orders()[field])
    np.testing.assert_allclose(pf.total_return(), explicit.total_return())
    assert pf.group_lens == explicit.group_lens
    assert pf.price_nbytes == close.nbytes
    np.testing.assert_array_equal(threshold(data)[0], original[0])


def test_run_and_bad_inputs():
    close = np.array([90.0, 110.0, 95.0, 120.0])
    pf = threshold.run({"close": close}, signal_delay=0, size=1)
    assert isinstance(pf, alphax.Portfolio)
    with pytest.raises(ValueError, match="cannot be empty"):
        threshold.sweep({"close": close}, params=[])
    with pytest.raises(TypeError, match="unexpected keyword"):
        threshold.run({"close": close}, params={"unknown": 1})
    with pytest.raises(ValueError, match="signal_delay"):
        threshold.run({"close": close}, signal_delay=-1)
    with pytest.raises(TypeError, match="managed"):
        threshold.run({"close": close}, column_map=[0])
    with pytest.raises(ValueError, match="expected"):
        @alphax.strategy
        def bad_shape(data):
            return np.array([True]), np.array([False])
        bad_shape.run({"close": close})
    with pytest.raises(TypeError, match="booleans"):
        @alphax.strategy
        def bad_type(data):
            return np.ones(len(close)), np.zeros(len(close))
        bad_type.run({"close": close})
