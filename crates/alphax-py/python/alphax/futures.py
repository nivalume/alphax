"""Futures data: continuous series across rolls, and contract specifications.

The engine trades one price series per column and has no notion of expiry.
A futures strategy therefore runs on a *continuous* series: the contract
held on each bar, stitched together at the rolls and adjusted so the jump
from one contract to the next is not a price move.

::

    held = alphax.futures.roll_by_volume(volume)            # or roll_by_expiry
    close_c = alphax.futures.continuous(close, held)          # back-adjusted
    rolls = alphax.futures.rolls(held)                        # bars a roll happens
    spec = alphax.futures.CONTRACTS["ES"]
    pf = alphax.from_signals(close_c, entries, exits, multiplier=spec.multiplier,
                             account_type="margin", initial_margin=12_000.0,
                             margin_basis="unit")

Back-adjusting keeps every price difference — and so every profit and loss in
contract points — exactly as traded, at the cost of earlier levels that no
contract ever traded at (possibly below zero: ``allow_negative_price``).
``"ratio"`` keeps returns instead of differences. The roll itself is not a
trade in the continuous series; a strategy that wants to pay for it trades
on the bars ``rolls`` marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CONTRACTS", "ContractSpec", "continuous", "roll_by_expiry", "roll_by_volume", "rolls", "spec_params"]


def _contracts(prices: Any, name: str) -> np.ndarray:
    array = np.asarray(prices, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be (n_bars, n_contracts); got {array.ndim} dimensions")
    return array


def _held(held: Any, n_rows: int, n_contracts: int) -> np.ndarray:
    held = np.asarray(held)
    if held.ndim != 1 or len(held) != n_rows or held.dtype.kind not in "iu":
        raise ValueError(f"held must be {n_rows} integer contract indices, one per bar")
    if held.size and (held.min() < 0 or held.max() >= n_contracts):
        raise ValueError(f"held must index the {n_contracts} contracts")
    return held.astype(np.intp)


def roll_by_volume(volume: Any) -> np.ndarray:
    """The contract held on each bar: the one with the most volume so far
    this bar, never rolling back to an earlier one.

    ``volume`` is ``(n_bars, n_contracts)``, contracts in expiry order, NaN
    where one does not trade. Bars before any volume hold the first contract.
    """
    volume = np.nan_to_num(_contracts(volume, "volume"), nan=-np.inf)
    most = np.argmax(volume, axis=1)
    return np.maximum.accumulate(most).astype(np.intp)


def roll_by_expiry(expiry_rows: Any, n_rows: int, *, before: int = 0) -> np.ndarray:
    """The contract held on each bar: each through ``before`` bars ahead of
    its expiry row (through the expiry row itself when 0), then the next.

    ``expiry_rows`` is each contract's last row, in expiry order. After the
    last contract's roll the last contract is held.
    """
    expiry = np.asarray(expiry_rows, dtype=np.int64)
    if expiry.ndim != 1 or (np.diff(expiry) <= 0).any():
        raise ValueError("expiry_rows must be increasing rows, one per contract")
    if before < 0:
        raise ValueError("before must be 0 or more")
    rows = np.arange(n_rows)
    held = np.searchsorted(expiry - before, rows, side="left")
    return np.minimum(held, len(expiry) - 1).astype(np.intp)


def rolls(held: Any) -> np.ndarray:
    """Whether the contract held changes on each bar."""
    held = np.asarray(held)
    out = np.zeros(len(held), dtype=bool)
    out[1:] = held[1:] != held[:-1]
    return out


def continuous(prices: Any, held: Any, *, method: str = "difference") -> np.ndarray:
    """One series from the contract held on each bar, adjusted at each roll.

    ``prices`` is ``(n_bars, n_contracts)`` and ``held`` the contract index of
    each bar. At a roll on bar ``t`` from contract ``a`` to ``b``, the gap is
    measured on that bar, ``prices[t, b]`` against ``prices[t, a]``, and every
    earlier bar is shifted by it: added (``"difference"``) or multiplied
    (``"ratio"``), so the series ends at the contract actually held. A roll
    on a bar where either contract has no price raises.
    """
    if method not in ("difference", "ratio"):
        raise ValueError(f"method must be 'difference' or 'ratio'; got {method!r}")
    prices = _contracts(prices, "prices")
    n_rows, n_contracts = prices.shape
    held = _held(held, n_rows, n_contracts)
    raw = prices[np.arange(n_rows), held]
    out = raw.copy()
    for t in np.flatnonzero(rolls(held))[::-1]:
        new, old = prices[t, held[t]], prices[t, held[t - 1]]
        if not (np.isfinite(new) and np.isfinite(old)):
            raise ValueError(f"the roll on bar {t} has no price for both contracts")
        if method == "difference":
            out[:t] += new - old
        else:
            if old == 0.0:
                raise ValueError(f"the roll on bar {t} is from a price of 0; use 'difference'")
            out[:t] *= new / old
    return out


@dataclass(frozen=True)
class ContractSpec:
    """A futures contract's terms that do not change: its ``multiplier``
    (cash per point), ``tick_size`` and ``currency``. Margins are set by the
    exchange and change, so they are the caller's."""

    multiplier: float
    tick_size: float
    currency: str = "USD"


#: Common contracts' terms, by root symbol, as the exchanges publish them.
CONTRACTS: dict[str, ContractSpec] = {
    "ES": ContractSpec(50.0, 0.25),
    "MES": ContractSpec(5.0, 0.25),
    "NQ": ContractSpec(20.0, 0.25),
    "MNQ": ContractSpec(2.0, 0.25),
    "YM": ContractSpec(5.0, 1.0),
    "RTY": ContractSpec(50.0, 0.1),
    "CL": ContractSpec(1000.0, 0.01),
    "MCL": ContractSpec(100.0, 0.01),
    "NG": ContractSpec(10000.0, 0.001),
    "GC": ContractSpec(100.0, 0.1),
    "MGC": ContractSpec(10.0, 0.1),
    "SI": ContractSpec(5000.0, 0.005),
    "HG": ContractSpec(25000.0, 0.0005),
    "ZN": ContractSpec(1000.0, 1 / 64),
    "ZB": ContractSpec(1000.0, 1 / 32),
    "ZF": ContractSpec(1000.0, 1 / 128),
    "ZC": ContractSpec(50.0, 0.25),
    "ZS": ContractSpec(50.0, 0.25),
    "ZW": ContractSpec(50.0, 0.25),
    "6E": ContractSpec(125000.0, 0.00005),
    "6J": ContractSpec(12500000.0, 0.0000005),
    "6B": ContractSpec(62500.0, 0.0001),
    "BTC": ContractSpec(5.0, 5.0),
    "MBT": ContractSpec(0.1, 5.0),
    "ETH": ContractSpec(50.0, 0.25),
    "FDAX": ContractSpec(25.0, 1.0, "EUR"),
    "FESX": ContractSpec(10.0, 1.0, "EUR"),
}


def spec_params(symbols: Any, specs: dict[str, ContractSpec] | None = None) -> dict[str, Any]:
    """``multiplier`` (and ``size_granularity`` of 1: whole contracts) for
    ``from_orders``/``from_signals``, one per column of ``symbols``."""
    table = CONTRACTS if specs is None else specs
    missing = [s for s in symbols if s not in table]
    if missing:
        raise KeyError(f"no contract spec for {missing}")
    return {
        "multiplier": np.array([table[s].multiplier for s in symbols]),
        "size_granularity": 1.0,
    }
