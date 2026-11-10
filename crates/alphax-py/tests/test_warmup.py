"""Lookbacks and warm-up through the Python API."""

from __future__ import annotations

import numpy as np
import alphax


def test_lookbacks_name_each_factor():
    factors = alphax.FactorSet.compile({"ma": "Mean($close, 20)", "lag": "Ref(Mean($close, 5), 2)", "raw": "$close"})
    assert factors.lookbacks == {"ma": 19, "lag": 6, "raw": 0}
    assert factors.lookback == 19
    assert alphax.alpha158().lookback == 60
    assert alphax.Expression.compile("Std($close, 10)").lookback == 9


def test_warmup_blanks_each_symbols_partial_windows():
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(size=(40, 3)), axis=0)
    close[:10, 2] = np.nan  # listed later
    factors = alphax.FactorSet.compile({"z": "($close - Mean($close, 5)) / Std($close, 5)"})
    warm = factors.with_warmup()
    assert warm.warmup and not factors.warmup
    raw = factors.evaluate({"close": close})["z"]
    blanked = warm.evaluate({"close": close})["z"]
    assert not np.isnan(raw[1:4, 0]).any()  # Qlib's partial windows
    for col, start in enumerate([0, 0, 10]):
        assert np.isnan(blanked[: start + 4, col]).all()
        np.testing.assert_array_equal(blanked[start + 4 :, col], raw[start + 4 :, col])
    # No entry can come from a partial window.
    entries = blanked > 1.0
    assert not entries[:4].any() and not entries[:14, 2].any()


def test_warmup_on_an_expression():
    close = np.arange(1.0, 11.0).reshape(-1, 1)
    values = alphax.Expression.compile("Mean($close, 3)").with_warmup().evaluate({"close": close})
    assert np.isnan(values[:2, 0]).all()
    np.testing.assert_array_equal(values[2:, 0], np.arange(2.0, 10.0))
