"""Prices quoted in another currency than the account's.

The engine keeps one cash balance per account, in one currency, and every
price must be in it. An instrument quoted in another currency is brought in
by converting its prices with the exchange rate of each bar:

::

    close_usd = alphax.fx.convert(close_eur, eurusd)          # EUR stock, USD account
    pf = alphax.from_signals(close_usd, entries, exits)

For an asset bought outright with account currency (a share, an ETF, a
coin in a cash column) this is exact: the position is worth its price times
the rate, and both the asset's move and the currency's are in the result.

For a contract settled in its own currency (a future or CFD in a margin
column) the converted series marks the whole notional at each bar's rate,
whereas the exchange pays the profit in the quote currency and only that
profit is exposed to the rate. The difference is the rate's move times the
position's notional, small over short holdings; the account then also holds
no foreign cash, as a multi-currency account would.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["convert", "cross"]


def convert(prices: Any, rate: Any, *, invert: bool = False) -> np.ndarray:
    """``prices`` in the account currency: times ``rate``, the account
    currency per unit of the quote currency (EURUSD for a EUR price in a USD
    account), or divided by it with ``invert`` (USDJPY for a JPY price).

    ``rate`` is a number, one per bar (1-D, for every column) or shaped like
    ``prices``. A NaN rate gives a NaN price.
    """
    prices = np.asarray(prices, dtype=np.float64)
    rate = np.asarray(rate, dtype=np.float64)
    if rate.ndim == 1 and prices.ndim == 2:
        rate = rate.reshape(-1, 1)
    if (rate <= 0).any():
        raise ValueError("an exchange rate must be above 0")
    return prices / rate if invert else prices * rate


def cross(base_usd: Any, quote_usd: Any) -> np.ndarray:
    """The rate of one currency in another through a third: ``base_usd /
    quote_usd`` (EURJPY from EURUSD and JPYUSD)."""
    return np.asarray(base_usd, dtype=np.float64) / np.asarray(quote_usd, dtype=np.float64)
