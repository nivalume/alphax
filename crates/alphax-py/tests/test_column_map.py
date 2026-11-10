"""Mapped source prices must match explicitly expanded prices throughout analysis."""
import gc
import numpy as np
import pytest
import alphax


@pytest.mark.parametrize("orders", [False, True])
@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("threads", [1, 4])
def test_mapping_matches_expansion_in_execution_and_analysis(orders, shared, threads):
    rng = np.random.default_rng(81)
    close = 100*np.exp(np.cumsum(rng.normal(0,0.02,(80,2)),axis=0))
    mapping = [1,0,1,0,1,0]
    entries = rng.random((80,6)) > .8
    exits = rng.random((80,6)) > .8
    kwargs = dict(threads=threads, log=True, fees=.001, init_cash=10000.)
    if shared:
        kwargs.update(cash_sharing=True, group_lens=[2,2,2], call_seq="auto")
    if orders:
        size = np.where(entries, .4, np.nan)
        kwargs.update(size_type="targetpercent")
        mapped = alphax.from_orders(close, size, column_map=mapping, **kwargs)
        expanded = alphax.from_orders(close[:,mapping], size, **kwargs)
    else:
        kwargs.update(sl_stop=.02, tp_stop=.04, sl_trail=True)
        mapped = alphax.from_signals(close, entries, exits, column_map=mapping,
            open=close, high=close*1.03, low=close*.97, **kwargs)
        expanded = alphax.from_signals(close[:,mapping], entries, exits,
            open=close[:,mapping], high=close[:,mapping]*1.03, low=close[:,mapping]*.97, **kwargs)
    assert mapped.price_nbytes == close.nbytes
    assert expanded.price_nbytes == close.nbytes*3
    # Portfolio owns one source copy: later mutation/release cannot corrupt stats.
    close[:] = -1000
    del close
    gc.collect()
    for method in ("orders", "logs", "trades", "metrics"):
        a, b = getattr(mapped, method)(), getattr(expanded, method)()
        assert a.keys() == b.keys()
        for field in a:
            np.testing.assert_allclose(a[field], b[field], equal_nan=True, err_msg=f"{method}.{field}")
    for i in range(mapped.n_cols):
        np.testing.assert_allclose(mapped.assets(col=i), expanded.assets(col=i))
    for i in range(mapped.n_groups):
        for method in ("cash", "value", "returns"):
            selector = {"group":i} if shared else {"col":i}
            np.testing.assert_allclose(getattr(mapped,method)(**selector), getattr(expanded,method)(**selector), equal_nan=True)


@pytest.mark.parametrize("mapping", [[],[-1],[2],[True],[0.0],[[0,1]]])
def test_invalid_maps(mapping):
    with pytest.raises((ValueError, TypeError, OverflowError)):
        alphax.from_orders(np.ones((3,2)), column_map=mapping)


def test_map_changes_column_order_even_when_widths_match():
    close = np.array([[10.,100.],[20.,90.],[15.,110.]])
    a = alphax.from_orders(close, [[1.,2.],[0.,0.],[-1.,-2.]], column_map=[1,0])
    b = alphax.from_orders(close[:,[1,0]], [[1.,2.],[0.,0.],[-1.,-2.]])
    np.testing.assert_allclose(a.total_return(), b.total_return())
