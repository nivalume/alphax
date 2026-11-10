"""Control cards for `alphax-lab calibrate`.

Negative controls (random factor, time-shuffled factor, future-return leak) must be
rejected; positive controls are real, well-known candidates whose outcome is only recorded.
"""

from __future__ import annotations

from alphax_lab.types.factor_ts import spec as specs

HORIZONS = [1, 4, 24]

_BASE_CARD = {
    "experiment_type": "factor_ts",
    "why_exists": "calibration control",
    "counterparty": "none",
    "expected_horizon": "days to weeks",
    "expected_behavior": {"majors": "control", "alts": "control"},
    "failure_conditions": "control",
    "direction": 1,
}

# The factor itself is replaced by the harness for negative controls; the grid gives members.
NEGATIVE_SPEC = specs.normalize(
    {
        "name": "negative_control",
        "expression": "Mean($close,{fast})/Mean($close,{slow})-1",
        "params": {"fast": [72, 168], "slow": [720, 1440]},
        "constraints": ["fast < slow"],
        "horizon": 24,
        "signal": {"rule": "sign", "side": "both", "params": {}},
    },
    HORIZONS,
)

LEAK_SPEC = {
    "name": "future_leak",
    "expression": "Ref($close,-24)/$close-1",
    "horizon": 24,
    "signal": {"rule": "sign", "side": "both", "params": {}},
}


def card(symbols: list[str], name: str) -> dict:
    return {
        **_BASE_CARD,
        "title": f"control: {name}",
        "mechanism": f"negative control ({name}); the pipeline must reject it",
        "symbols": symbols,
        "param_ranges": {"fast": [24, 720], "slow": [168, 2160]},
        "source": {"kind": "calibration"},
    }


def positives(symbols: list[str]) -> dict[str, tuple[dict, dict]]:
    """Well-known candidates with textbook parameter ranges, fixed before any run."""
    items = {
        "ma_trend": (
            {
                "title": "control: moving-average trend",
                "mechanism": "slow diffusion of information and herding make multi-week price moves persist",
                "direction": 1,
                "param_ranges": {"fast": [72, 336], "slow": [720, 2160]},
            },
            {
                "name": "ma_trend",
                "expression": "Mean($close,{fast})/Mean($close,{slow})-1",
                "params": {"fast": [72, 168, 336], "slow": [720, 1440, 2160]},
                "horizon": 24,
                "signal": {"rule": "sign", "side": "both", "params": {}},
            },
        ),
        "short_reversal": (
            {
                "title": "control: short-term reversal",
                "mechanism": "liquidity demand pushes prices away from value for hours; liquidity providers are paid to revert it",
                "direction": -1,
                "param_ranges": {"n": [4, 24], "window": [168, 720], "scale": [1.0, 2.0]},
            },
            {
                "name": "short_reversal",
                "expression": "$close/Ref($close,{n})-1",
                "params": {"n": [4, 12, 24]},
                "horizon": 4,
                "signal": {"rule": "zscore", "side": "both", "params": {"window": [168, 720], "scale": [1.0, 2.0]}},
            },
        ),
        "range_breakout": (
            {
                "title": "control: range breakout",
                "mechanism": "stop orders and late trend followers cluster beyond recent extremes, extending breakouts",
                "direction": 1,
                "param_ranges": {"n": [168, 2160]},
            },
            {
                "name": "range_breakout",
                "expression": "($close-Min($low,{n}))/(Max($high,{n})-Min($low,{n})+1e-12)-0.5",
                "params": {"n": [168, 336, 720, 1440, 2160]},
                "horizon": 24,
                "signal": {"rule": "sign", "side": "both", "params": {}},
            },
        ),
    }
    out = {}
    for name, (card_fields, raw_spec) in items.items():
        full_card = {**_BASE_CARD, **card_fields, "symbols": symbols, "source": {"kind": "calibration"},
                     "why_exists": "textbook positive control", "failure_conditions": "regime without persistent moves"}
        out[name] = (full_card, specs.normalize(raw_spec, HORIZONS))
    return out
