"""Protocol behaviour with the experiment type's runs replaced by fixtures (no market data needed)."""

import json

import pytest
import yaml
from alphax_lab import pipeline
from alphax_lab.protocol import experiment_type
from alphax_lab.types.factor_ts import FactorTS

from tests.test_spec_and_cards import CARD, MA_GAP


def _result(t: float, members: float = 0.8, cross: float = 1.0, sharpe_2x: float = 0.5) -> dict:
    return {
        "net": {"t_stat": t, "sharpe": t / 2, "days": 1461, "annualized_return": 0.1, "max_drawdown": -0.3,
                "orders": 10, "turnover": 5.0, "fees": 100.0},
        "sharpe_2x": sharpe_2x,
        "cost_sensitivity": {"0x": 1.0, "1x": 0.9, "2x": sharpe_2x},
        "by_year_sharpe": {"2021": 1.0},
        "daily_returns_1x": [0.001, -0.0005, 0.002] * 100,
        "members": {"sharpe": [1.0], "positive_share": members, "params": [{}]},
        "cross_asset": {"by_symbol": {}, "positive_share": cross},
        "selection": {"method": "ensemble"},
    }


class FakeType(FactorTS):
    def __init__(self, results: dict):
        self.results = results

    def contract_check(self, spec, card, policy):
        return None

    def run(self, spec, card, policy, window, *, control=None):
        return self.results[(spec["name"], window.split(":")[0])]


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    results = {}
    monkeypatch.setattr(experiment_type, "get", lambda name: FakeType(results))
    lab = pipeline.Lab(tmp_path / "ledger")
    lab.results = results
    return lab


def _pending(lab, name: str, expression: str, title: str) -> str:
    item = {"id": name, "card": dict(CARD, title=title, symbols=["BTCUSDT"]),
            "spec": dict(MA_GAP, name=name, expression=expression)}
    (lab.root / "pending" / f"{name}.yaml").write_text(yaml.safe_dump(item))
    return name


def _register(lab, name, expression, title):
    return lab.approve(_pending(lab, name, expression, title))["spec_id"]


def test_register_run_once_and_count(lab):
    sid = _register(lab, "a", MA_GAP["expression"], "trend a")
    lab.results[("a", "in_sample")] = _result(t=2.5)
    [entry] = lab.run()
    assert entry["passed"] and entry["tests_at_run"] == 1
    assert lab.run() == []  # never twice
    assert (lab.root / "reports" / f"{sid}.in_sample.md").read_text().startswith("# trend a — PASSED")
    with pytest.raises(ValueError, match="same structure"):
        _register(lab, "a2", "Mean($vwap,{fast})/Mean($vwap,{slow})-1", "trend a again")


def test_threshold_rises_with_the_count(lab):
    lab.ledger.append("legacy", {"source": "x", "tests": 9, "oos_tests": 0})
    _register(lab, "b", "Mean($high,{fast})/Mean($low,{slow})-1*Std($volume,24)", "b")
    lab.results[("b", "in_sample")] = _result(t=2.5)
    [entry] = lab.run()
    assert entry["tests_at_run"] == 10 and not entry["passed"]  # 2.5 < 2.81


@pytest.mark.parametrize(
    "kwargs, failing",
    [({"sharpe_2x": -0.1}, "sharpe_at_2x_cost"), ({"members": 0.5}, "grid_members_positive"),
     ({"cross": 0.5}, "cross_asset_positive"), ({"t": 9.0}, "not_suspicious")],
)
def test_each_gate_can_reject(lab, kwargs, failing):
    _register(lab, "c", "Std($close,{fast})/Std($close,{slow})", "c")
    lab.results[("c", "in_sample")] = _result(**({"t": 3.0} | kwargs))
    [entry] = lab.run()
    assert [c["name"] for c in entry["gate"] if not c["passed"]] == [failing]


def test_policy_change_makes_passes_stale(lab, tmp_path):
    _register(lab, "d", MA_GAP["expression"], "d")
    lab.results[("d", "in_sample")] = _result(t=3.0)
    lab.run()
    policy = yaml.safe_load(lab.policy_path.read_text())
    policy["gate"]["alpha"] = 0.1
    lab.policy_path.write_text(yaml.safe_dump(policy))
    reopened = pipeline.Lab(lab.root)
    assert reopened.oos_candidates() == []
    assert reopened.status()[0]["stage"].startswith("stale")


def test_oos_batches_are_spaced_and_limited(lab):
    _register(lab, "e", MA_GAP["expression"], "e")
    lab.results[("e", "in_sample")] = _result(t=3.0)
    lab.results[("e", "out_of_sample")] = _result(t=2.0)
    lab.run()
    batch = lab.oos()
    assert batch["tests_at_batch"] == 1 and batch["results"][batch["spec_ids"][0]]["passed"]
    _register(lab, "f", "Std($close,{fast})/Std($close,{slow})", "f")
    lab.results[("f", "in_sample")] = _result(t=3.0)
    lab.results[("f", "out_of_sample")] = _result(t=2.0)
    lab.run()
    with pytest.raises(RuntimeError, match="next OOS batch is due"):
        lab.oos()
    second = lab.oos(force=True)
    assert second["forced"] and second["tests_at_batch"] == 2
    lab.ledger.append("oos_batch", {"spec_ids": [], "results": {}, "forced": True})
    with pytest.raises(RuntimeError, match="used up"):
        lab.oos(force=True)


def test_forward_retires_on_deviation(lab):
    _register(lab, "g", MA_GAP["expression"], "g")
    lab.results[("g", "in_sample")] = _result(t=3.0)
    lab.results[("g", "out_of_sample")] = _result(t=2.0)
    lab.run()
    lab.oos()
    bad = _result(t=-3.0)
    bad["daily_returns_1x"] = [-0.01] * 40
    lab.results[("g", "forward")] = bad
    [row] = lab.forward()
    assert row["status"] == "retired"
    assert lab.forward() == []


class RecordingLLM:
    def __init__(self, replies):
        self.replies, self.transcript = list(replies), []

    def chat_json(self, system, user, attempts=3):
        self.transcript.append({"system": system, "user": user})
        return self.replies.pop(0)


def test_proposals_never_see_results(lab):
    _register(lab, "h", MA_GAP["expression"], "trend h")
    lab.results[("h", "in_sample")] = _result(t=3.0)
    lab.run()
    good = {"card": dict(CARD, title="volume shock"), "spec": dict(MA_GAP, name="vol_shock",
            expression="Mean($volume,{fast})/Mean($volume,{slow})-1")}
    dated = {"card": dict(CARD, title="old story", mechanism="worked well in 2021"), "spec": MA_GAP}
    fixed = {"card": dict(CARD, title="old story fixed"), "spec": dict(MA_GAP, name="ret_gap",
             expression="Mean($close/Ref($close,1),{fast})-Mean($close/Ref($close,1),{slow})")}
    llm = RecordingLLM([{"cards": [good, dated]}, fixed])
    items = lab.propose(llm, symbols=["BTCUSDT"], n=2, prompt="volume leads price")
    assert [i["problems"] for i in items] == [[], []]
    assert items[1]["card"]["title"] == "old story fixed"
    prompts = " ".join(t["system"] + t["user"] for t in llm.transcript).lower()
    assert "trend h" in prompts  # de-duplication sees titles...
    for word in ("3.0", "passed", "t_stat", "sharpe_2x", "tests_at_run"):  # ...but no results
        assert word not in prompts, word
    assert "2021" in llm.transcript[1]["user"]  # only the rejected card's own text went back for a fix


def test_real_catalog_run(tmp_path):
    """End to end on real bars with the real experiment type (skipped without the catalog)."""
    from pathlib import Path

    from alphax_lab import data as bars

    policy = pipeline.policies.DEFAULT_POLICY
    key = bars.bar_key("BTCUSDT", policy["data"]["venue"], policy["data"]["interval"])
    if not (Path(policy["data"]["catalog"]).expanduser() / "data" / "bars" / key).exists():
        pytest.skip("alphax catalog with BTCUSDT bars is not available")
    lab = pipeline.Lab(tmp_path / "ledger")
    policy_doc = yaml.safe_load(lab.policy_path.read_text())
    policy_doc["harness"]["alpha158_reference"] = False
    lab.policy_path.write_text(yaml.safe_dump(policy_doc))
    lab = pipeline.Lab(tmp_path / "ledger")
    sid = _register(lab, "trend", MA_GAP["expression"], "trend")
    [entry] = lab.run()
    assert "error" not in entry, entry.get("traceback")
    result = json.loads((lab.root / "reports" / f"{sid}.in_sample.json").read_text())
    assert result["window"]["end"] == "2025-01-01" and result["window"]["bars"] > 30000
    assert set(result["cross_asset"]["by_symbol"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}
    assert len(result["members"]["sharpe"]) == 4
