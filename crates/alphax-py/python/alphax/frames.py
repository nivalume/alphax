"""Results as pandas objects. Needs pandas (``pip install alphax-trader[pandas]``).

Records come back as DataFrames, series as DataFrames with one column per
account, both with the bars' ``index`` when given (the ``idx`` columns of
records stay row numbers; ``*_time`` columns are added beside them)::

    alphax.frames.orders(pf, index=dates)
    alphax.frames.value(pf, index=dates)
    alphax.frames.stats(pf)          # one column per account
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["drawdowns", "orders", "records", "returns", "stats", "trades", "value"]


def _pd():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("alphax.frames needs pandas: pip install pandas") from e
    return pd


def records(rec: dict, index: Any = None) -> Any:
    """A records dict as a DataFrame; each ``*idx`` column gets a ``*time``
    column from ``index``."""
    pd = _pd()
    df = pd.DataFrame({k: np.asarray(v) for k, v in rec.items()})
    if index is not None:
        index = np.asarray(index)
        for col in [c for c in df.columns if c == "idx" or c.endswith("_idx")]:
            rows = df[col].to_numpy()
            valid = (rows >= 0) & (rows < len(index))
            times = np.full(len(rows), np.datetime64("NaT"), dtype=index.dtype) if index.dtype.kind == "M" \
                else np.full(len(rows), None, dtype=object)
            times[valid] = index[rows[valid]]
            df[col[:-3] + "time" if col.endswith("_idx") else "time"] = times
    return df


def orders(pf: Any, index: Any = None) -> Any:
    """The fills."""
    return records(pf.orders(), index)


def trades(pf: Any, index: Any = None) -> Any:
    """The trades, in the portfolio's trade view."""
    return records(pf.trades(), index)


def drawdowns(pf: Any, index: Any = None) -> Any:
    """The drawdown episodes."""
    return records(pf.drawdowns(), index)


def _accounts(pf: Any) -> list:
    return list(range(len(pf.group_lens))) if pf.cash_sharing else list(range(pf.n_groups))


def _series(pf: Any, method: str, index: Any) -> Any:
    pd = _pd()
    data = {a: (getattr(pf, method)(group=a) if pf.cash_sharing else getattr(pf, method)(a))
            for a in _accounts(pf)}
    return pd.DataFrame(data, index=index)


def value(pf: Any, index: Any = None) -> Any:
    """Each account's value per bar."""
    return _series(pf, "value", index)


def returns(pf: Any, index: Any = None) -> Any:
    """Each account's return per bar."""
    return _series(pf, "returns", index)


def stats(pf: Any, **kwargs: Any) -> Any:
    """The statistics table, one column per account; ``kwargs`` go to
    ``pf.stats`` (``high=``, ``low=``)."""
    pd = _pd()
    cols = {a: (pf.stats(group=a, **kwargs) if pf.cash_sharing else pf.stats(a, **kwargs))
            for a in _accounts(pf)}
    return pd.DataFrame(cols)
