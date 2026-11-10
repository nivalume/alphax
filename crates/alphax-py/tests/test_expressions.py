"""Factor-expression Python bindings."""

import alphax
import numpy as np
import pytest


def test_expression_compiles_once_and_evaluates_custom_fields():
    expression = alphax.Expression.compile("$funding_rate - Mean($funding_rate, 2)")
    first = expression.evaluate({"funding_rate": [1.0, 3.0, 5.0]})
    second = expression.evaluate({"funding_rate": [2.0, 4.0, 8.0]})

    np.testing.assert_allclose(first[:, 0], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(second[:, 0], [0.0, 1.0, 2.0])
    assert first.shape == (3, 1)
    assert first.flags.f_contiguous
    assert expression.required_fields == ("funding_rate",)
    assert not expression.uses_future


def test_factor_set_preserves_names_and_computes_multiple_symbols():
    factors = alphax.FactorSet.compile(
        {
            "return": "$close / Ref($close, 1) - 1",
            "mean_ratio": "Mean($close, 2) / $close",
        }
    )
    close = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
    values = factors.evaluate({"close": close})

    assert factors.names == ("return", "mean_ratio")
    assert factors.required_fields == ("close",)
    assert tuple(values) == factors.names
    np.testing.assert_allclose(values["return"][1:], 1.0)
    np.testing.assert_allclose(
        values["mean_ratio"], [[1.0, 1.0], [0.75, 0.75], [0.75, 0.75]]
    )
    assert all(value.flags.f_contiguous for value in values.values())


def test_inputs_accept_c_order_and_non_contiguous_slices():
    source = np.arange(24.0).reshape(4, 6)
    result = alphax.Expression.compile("$x + 1").evaluate({"x": source[:, ::2]})

    np.testing.assert_array_equal(result, source[:, ::2] + 1)
    assert result.flags.f_contiguous


def test_expression_metadata_and_errors_are_explicit():
    assert alphax.Expression.compile("Ref($close, -1)").uses_future
    with pytest.raises(ValueError, match="unsupported operator"):
        alphax.Expression.compile("Magic($close)")
    with pytest.raises(ValueError, match=r"missing field \$volume"):
        alphax.Expression.compile("$close + $volume").evaluate({"close": [1.0]})
    with pytest.raises(ValueError, match="has shape"):
        alphax.Expression.compile("$close + $volume").evaluate(
            {"close": [1.0, 2.0], "volume": [1.0]}
        )
    with pytest.raises(ValueError, match="duplicate factor field"):
        alphax.Expression.compile("$close").evaluate({"close": [1.0], "CLOSE": [1.0]})
    with pytest.raises(ValueError, match="cannot be empty"):
        alphax.Expression.compile("$close").evaluate({})
    with pytest.raises(ValueError, match="non-zero"):
        alphax.Expression.compile("$close").evaluate({"close": []})
    with pytest.raises(TypeError, match="mapping"):
        alphax.FactorSet.compile([("x", "$close")])


def test_alpha158_requires_vwap_and_returns_owned_arrays():
    factors = alphax.alpha158()
    base = np.arange(1.0, 9.0).reshape(4, 2)
    fields = {
        "open": base,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.25,
        "volume": base * 100.0,
        "vwap": base + 0.1,
    }

    with pytest.raises(ValueError, match=r"missing field \$vwap"):
        factors.evaluate(
            {name: value for name, value in fields.items() if name != "vwap"}
        )

    values = factors.evaluate(fields)
    assert len(values) == 158
    assert tuple(values) == factors.names
    assert all(value.shape == (4, 2) for value in values.values())
    del fields, base
    assert values["KMID"].shape == (4, 2)


def test_alpha360_matches_qlib_order_and_is_reusable():
    factors = alphax.alpha360()
    expected_names = tuple(
        f"{field}{offset}"
        for field in ("CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME")
        for offset in range(59, -1, -1)
    )
    base = np.arange(1.0, 129.0).reshape(64, 2)
    fields = {
        "open": base + 0.1,
        "high": base + 1.0,
        "low": base - 1.0,
        "close": base + 0.25,
        "volume": base * 100.0,
        "vwap": base + 0.2,
    }

    assert factors.names == expected_names
    assert factors.required_fields == ("close", "high", "low", "open", "volume", "vwap")
    assert not factors.uses_future
    with pytest.raises(ValueError, match=r"missing field \$vwap"):
        factors.evaluate(
            {name: value for name, value in fields.items() if name != "vwap"}
        )

    first = factors.evaluate(fields)
    second = factors.evaluate({name: value + 1.0 for name, value in fields.items()})
    assert tuple(first) == expected_names
    assert all(value.shape == (64, 2) for value in first.values())
    assert all(value.flags.f_contiguous for value in first.values())
    assert all(
        not np.shares_memory(value, source)
        for value in first.values()
        for source in fields.values()
    )
    assert all(
        not np.shares_memory(first[name], second[name]) for name in expected_names
    )
    assert np.isnan(first["CLOSE59"][58, 0])
    np.testing.assert_allclose(
        first["CLOSE59"][59], fields["close"][0] / fields["close"][59]
    )
    np.testing.assert_allclose(first["CLOSE0"], 1.0)
