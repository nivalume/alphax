"""The gate policy: every threshold and period the protocol judges by, versioned by content hash.

The policy file lives in the ledger repository. Editing it changes `policy_version`,
which makes every earlier pass stale (see report.py): reading results and then moving
the goalposts is the same feedback edge the protocol exists to cut.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import yaml

DEFAULT_POLICY: dict = {
    "periods": {
        # In-sample is the only data `run` may load. OOS is 2025 onward: out of sample for the
        # pipeline, NOT for an LLM whose training data covers 2025.
        "in_sample": ["2021-01-01", "2025-01-01"],
        "out_of_sample": ["2025-01-01", "2026-09-01"],
    },
    "data": {
        "catalog": "~/.cache/alphax/catalog",
        "venue": "BINANCE",
        "interval": "1h",
        # Cross-asset robustness universe. Correlated around 0.8, so it is a robustness
        # check, not extra sample.
        "universe": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"],
    },
    "costs": {"init_cash": 100000.0, "fees": 0.001, "slippage": 0.0005, "funding_modeled": False},
    "harness": {
        "position_step": 0.25,
        "normalize_window": 720,
        "horizons": [1, 4, 24],
        "placebos": 20,
        "alpha158_reference": True,
        "seed": 20260922,
    },
    "spec_limits": {"max_grid": 64, "max_nodes": 40, "max_free_ratio": 0.5, "max_common_subtree": 8},
    "gate": {
        # t of daily net returns must reach the Bonferroni bound for the ledger's global count
        # of in-sample tests N: 1.96 at N=1, 2.81 at N=10, 3.29 at N=50.
        "alpha": 0.05,
        "min_sharpe_at_2x_cost": 0.0,
        "min_member_positive_share": 0.6,
        "min_cross_asset_positive_share": 0.75,
        # Beyond this the result is treated as a look-ahead bug, not an edge.
        "suspicious_t": 8.0,
    },
    "oos": {"alpha": 0.05, "min_days_between_batches": 90, "max_batches": 3},
    "forward": {"interim_days": 90, "final_days": 180, "retire_z": -2.0},
    "card": {
        # Hypothesis cards describe mechanisms, never when something happened.
        "forbidden_terms": [
            "halving", "etf", "ftx", "luna", "terra", "celsius", "3ac", "three arrows", "covid", "election",
            "fomc", "merge", "shanghai upgrade", "mt. gox", "mt gox", "bitmex", "binance ban", "china ban",
        ],
    },
}


def t_required(tests: int, alpha: float, two_sided: bool = True) -> float:
    """Bonferroni bound: the t a single test must reach when `tests` tests share `alpha`."""
    tests = max(int(tests), 1)
    tail = alpha / (2 * tests) if two_sided else alpha / tests
    return NormalDist().inv_cdf(1 - tail)


def load(path: Path) -> dict:
    if not path.exists():
        path.write_text(yaml.safe_dump(DEFAULT_POLICY, sort_keys=False, allow_unicode=True))
    return yaml.safe_load(path.read_text())


def version(policy: dict) -> str:
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]
