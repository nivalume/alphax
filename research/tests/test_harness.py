import numpy as np
import pytest

pytest.importorskip("alphax")

from alphax_lab import data as bars  # noqa: E402
from alphax_lab.protocol.policy import DEFAULT_POLICY  # noqa: E402
from alphax_lab.types.factor_ts import harness  # noqa: E402
from alphax_lab.types.factor_ts import spec as fs  # noqa: E402

COSTS = DEFAULT_POLICY["costs"]


def test_labels_start_at_the_execution_bar():
    close = np.arange(1.0, 11.0)[:, None]
    labels = harness.labels(close, 2)
    assert labels[0, 0] == pytest.approx(close[3, 0] / close[1, 0] - 1)
    assert np.isnan(labels[-3:]).all()


def test_ic_detects_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    rows = 24 * 400
    time = np.arange(rows, dtype=np.int64) * bars.HOUR_NS + bars.time_ns("2021-01-01")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (rows, 2)), axis=0))
    label = harness.labels(close, 1)
    months = harness.months_of(time)
    assert harness.ic_summary(harness.ic_table(label + rng.normal(0, 0.02, label.shape), label, months)[1])["mean_ic"] > 0.3
    assert abs(harness.ic_summary(harness.ic_table(rng.normal(0, 1, label.shape), label, months)[1])["mean_ic"]) < 0.03


def test_normalized_ic_removes_the_persistent_level_bias():
    """A random walk's log price cannot predict returns, but its raw IC is consistently negative."""
    raw, normalized = [], []
    for seed in range(1, 7):
        rng = np.random.default_rng(seed)
        rows = 24 * 1000
        time = np.arange(rows, dtype=np.int64) * bars.HOUR_NS + bars.time_ns("2021-01-01")
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (rows, 1)), axis=0))
        label, months = harness.labels(close, 24), harness.months_of(time)
        level = np.log(close)
        raw.append(harness.ic_summary(harness.ic_table(level, label, months)[1])["mean_ic"])
        z = harness.zscore([level], 720)[0]
        normalized.append(harness.ic_summary(harness.ic_table(z, label, months)[1])["mean_ic"])
    assert max(raw) < -0.01
    assert abs(np.mean(normalized)) < 0.03


def test_targets_only_trade_on_position_changes():
    positions = np.array([[0.0], [0.4], [0.6], [0.55], [0.0], [0.0]])
    targets = harness.targets(positions, 0.25, 1)
    assert np.isnan(targets[0, 0]) and targets[1, 0] == 0.0 and targets[2, 0] == 0.5
    assert np.isnan(targets[3:5, 0]).all() and targets[5, 0] == 0.0


def test_band_state_holds_until_exit():
    z = np.array([[0.0], [1.5], [0.8], [0.2], [-0.1], [-1.2], [-0.5], [0.3]])
    position = harness.position(z, z, "band", {"entry": 1.0, "exit": 0.0}, 1, "both")
    assert position[:, 0].tolist() == [0, 1, 1, 1, 0, -1, -1, 0]


def test_direction_is_applied_not_searched():
    factor = np.array([[1.0], [-2.0], [0.5]])
    assert harness.position(factor, None, "sign", {}, -1, "both")[:, 0].tolist() == [-1, 1, -1]


def _spec(selection):
    return fs.normalize(
        {
            "name": "mom",
            "expression": "$close/Ref($close,{n})-1",
            "params": {"n": [24, 72, 168]},
            "horizon": 24,
            "signal": {"rule": "sign"},
            "selection": selection,
        },
        [1, 4, 24],
    )


@pytest.mark.parametrize(
    "selection",
    [{"method": "ensemble"}, {"method": "walk_forward", "metric": "sharpe", "lookback_bars": 720, "refit_bars": 168}],
)
def test_composite_is_causal(selection):
    """Appending later bars must not change earlier composite positions (walk-forward uses only the past)."""
    spec = _spec(selection)
    panel = bars.synthetic(3, rows=3000, cols=1)
    fields = {k: panel[k] for k in bars.FIELDS}
    full, _ = harness.composite(spec, harness.member_positions(spec, *harness.member_factors(spec, fields), 1), panel["close"], COSTS, 0.25)
    cut = 2000
    prefix = {k: v[:cut] for k, v in fields.items()}
    early, info = harness.composite(spec, harness.member_positions(spec, *harness.member_factors(spec, prefix), 1), panel["close"][:cut], COSTS, 0.25)
    np.testing.assert_allclose(early, full[:cut])
    if selection["method"] == "walk_forward":
        assert info["refits"] > 0 and not early[:720].any()


def test_contract_check_reports_bad_values():
    broken = fs.normalize(
        {"name": "b", "expression": "Mean($close,{n})/($volume-$volume)", "params": {"n": [5]}, "horizon": 24,
         "signal": {"rule": "sign"}},
        [1, 4, 24],
    )
    assert "NaN" in harness.contract_check(broken, 1, 7, 0.25, COSTS)
    assert harness.contract_check(_spec({"method": "ensemble"}), 1, 7, 0.25, COSTS) is None


def test_daily_t_uses_day_boundaries():
    time = np.arange(24 * 10, dtype=np.int64) * bars.HOUR_NS + bars.time_ns("2021-01-01")
    values = 100 * np.exp(np.linspace(0, 0.1, time.size))
    assert harness.daily_returns(values, time).size == 10
