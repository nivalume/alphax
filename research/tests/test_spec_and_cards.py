import pytest
from alphax_lab.protocol import cards, policy
from alphax_lab.types.factor_ts import spec as fs

HORIZONS = [1, 4, 24]
LIMITS = dict(horizons=HORIZONS, max_grid=64, max_nodes=40, max_free_ratio=0.5, max_common_subtree=8, others={})
MA_GAP = {
    "name": "ma_gap",
    "expression": "Mean($close,{fast})/Mean($close,{slow})-1",
    "params": {"fast": [168, 72], "slow": [720, 2160]},
    "constraints": ["fast < slow"],
    "horizon": 24,
    "signal": {"rule": "sign"},
}
CARD = {
    "title": "multi-week trend",
    "experiment_type": "factor_ts",
    "mechanism": "slow information diffusion makes multi-week moves persist",
    "why_exists": "trend followers are paid for bearing crash risk",
    "counterparty": "value traders fading moves too early",
    "expected_horizon": "weeks",
    "expected_behavior": {"majors": "weak but positive", "alts": "stronger"},
    "failure_conditions": "range-bound markets",
    "direction": 1,
    "param_ranges": {"fast": [24, 336], "slow": [720, 2160]},
}


def test_expand_and_selection_defaults():
    spec = fs.normalize(MA_GAP, HORIZONS)
    assert spec["params"]["fast"] == [72.0, 168.0]
    assert spec["selection"] == {"method": "ensemble"}
    assert len(fs.expand(spec)) == 4


def test_walk_forward_selection_is_validated():
    ok = fs.normalize(dict(MA_GAP, selection={"method": "walk_forward", "lookback_bars": 2160, "refit_bars": 168}), HORIZONS)
    assert ok["selection"]["metric"] == "sharpe"
    with pytest.raises(fs.SpecError, match="lookback_bars"):
        fs.normalize(dict(MA_GAP, selection={"method": "walk_forward", "lookback_bars": 10, "refit_bars": 168}), HORIZONS)


@pytest.mark.parametrize(
    "expression, message",
    [
        ("Ref($close,-1)/$close", "future"),
        ("Magic($close)", "unsupported operator"),
        ("Mean($close)", "takes 2 arguments"),
        ("$funding/$close", "unknown field"),
        ("Mean($close,{n}", r"expected '\)'"),
    ],
)
def test_parse_rejects_invalid_expressions(expression, message):
    with pytest.raises(fs.SpecError, match=message):
        fs.parse(expression)


def test_parse_matches_alphax_grammar():
    alphax = pytest.importorskip("alphax")
    for source in ("($close-Ref($close,1))/(2*$close+1e-12)", "If($close>Mean($close,20),1,-1)", "Abs($volume)**0.5"):
        fs.parse(source)
        alphax.Expression.compile(source)


def test_values_must_lie_in_card_ranges():
    outside = dict(MA_GAP, params={"fast": [12, 72], "slow": [720]})
    _, problems = fs.regulate(outside, **LIMITS, ranges=CARD["param_ranges"])
    assert any("outside the card range" in p for p in problems)
    unranged = dict(MA_GAP, signal={"rule": "zscore", "params": {"window": [720], "scale": [1.0]}})
    _, problems = fs.regulate(unranged, **LIMITS, ranges=CARD["param_ranges"])
    assert any("no range" in p for p in problems)


def test_resweeping_or_swapping_price_field_is_not_new():
    tried = {"ma_gap": MA_GAP["expression"]}
    for variant in (MA_GAP["expression"], "Mean($vwap,{fast})/Mean($vwap,{slow})-1"):
        _, problems = fs.regulate(dict(MA_GAP, name="again", expression=variant), **LIMITS, tried=tried)
        assert any("same structure" in p for p in problems)
    filtered = dict(MA_GAP, name="filtered", expression="(Mean($close,{fast})/Mean($close,{slow})-1)*Std($volume,24)")
    assert fs.regulate(filtered, **LIMITS, tried=tried)[1] == []


def test_valid_card_passes():
    assert cards.regulate(CARD, policy.DEFAULT_POLICY["card"]["forbidden_terms"]) == []


@pytest.mark.parametrize(
    "field, text, fragment",
    [
        ("mechanism", "trends persisted after the 2021 rally", "a year"),
        ("mechanism", "since 2024-03 funding stayed positive", "a date"),
        ("why_exists", "as seen in March 12 liquidations", "a month"),
        ("mechanism", "after the FTX collapse liquidity thinned", "ftx"),
        ("counterparty", "buyers chasing the halving narrative", "halving"),
        ("mechanism", "趋势在2022年熊市中持续", "a year"),
    ],
)
def test_card_rejects_dates_and_events(field, text, fragment):
    problems = cards.regulate(dict(CARD, **{field: text}), policy.DEFAULT_POLICY["card"]["forbidden_terms"])
    assert any(fragment in p.lower() for p in problems), problems


def test_card_requires_fixed_direction_and_ranges():
    problems = cards.regulate(dict(CARD, direction=0, param_ranges={}), [])
    assert any("direction" in p for p in problems) and any("param_ranges" in p for p in problems)


def test_bonferroni_thresholds():
    assert policy.t_required(1, 0.05) == pytest.approx(1.96, abs=0.01)
    assert policy.t_required(10, 0.05) == pytest.approx(2.81, abs=0.01)
    assert policy.t_required(50, 0.05) == pytest.approx(3.29, abs=0.01)
    assert policy.t_required(1, 0.05, two_sided=False) == pytest.approx(1.645, abs=0.01)
