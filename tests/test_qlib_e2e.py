"""End to end on qlib: export, coding search, look-ahead check, backtests, promotion, de-duplication.

Runs the real qlib workflow on a synthetic binary dataset; only the language model is scripted.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from support import GOOD_FACTOR, KEY_ERROR_FACTOR, ScriptedBackend, SyntheticMarket

from alphax_agent.app import AlphaXApplication
from alphax_agent.config.settings import AlphaXSettings, CoSTEERSettings, LLMSettings, WorkflowSettings
from alphax_agent.core.promotion import MetricCriterion, PromotionRule
from alphax_agent.llm.client import LLMClient
from alphax_agent.scenarios.qlib_factor import QlibFactorScenario, QlibFactorSettings

pytestmark = [
    pytest.mark.qlib,
    pytest.mark.skipif(importlib.util.find_spec("qlib") is None, reason="qlib is not installed"),
]

DOUBLED_FACTOR = GOOD_FACTOR.replace("factor = -returns", "factor = -2 * returns")


class ResearchScript:
    """Scripted replies: loop 0 tests 5-day reversal (first draft crashes), loop 1 a rescaled copy of it."""

    def __init__(self) -> None:
        self.designs = 0

    def install(self, backend: ScriptedBackend) -> None:
        backend.on("factor_proposer", self.proposal)
        backend.on("factor_designer", self.design)
        backend.on("factor_coder", self.code)
        backend.on(
            "factor_critic",
            lambda request: json.dumps({"consistent": True, "score": 0.9, "critique": "Implements the formulation."}),
        )
        backend.on(
            "reflector",
            lambda request: json.dumps(
                {
                    "lessons": [
                        {"text": "Columns keep the $ prefix: read df['$close'].", "signature": "execution:KeyError"}
                    ],
                    "reinforced": [],
                }
            ),
        )
        backend.on(
            "factor_analyst",
            lambda request: json.dumps(
                {
                    "observations": "IC moved.",
                    "hypothesis_evaluation": "Consistent with reversal.",
                    "hypothesis_supported": True,
                    "next_hypothesis": "Try volume-conditioned reversal.",
                    "reason": "See metrics.",
                }
            ),
        )

    @staticmethod
    def proposal(request) -> str:
        return json.dumps(
            {
                "statement": "Short-term losers outperform over the next days.",
                "rationale": "Liquidity providers are paid for absorbing order imbalance.",
            }
        )

    def design(self, request) -> str:
        self.designs += 1
        name = "REV5" if self.designs == 1 else "REV5_X2"
        return json.dumps(
            {
                "factors": [
                    {
                        "name": name,
                        "description": "negative mean 5-day return",
                        "formulation": r"-\frac{1}{5}\sum_{i=0}^{4} r_{t-i}",
                    }
                ]
            }
        )

    @staticmethod
    def code(request) -> str:
        user = request.messages[1].content
        name = re.search(r"Factor name: (\w+)", user).group(1)
        repaired = GOOD_FACTOR if "Mode: repair" in user else KEY_ERROR_FACTOR
        template = DOUBLED_FACTOR if name == "REV5_X2" else repaired
        return json.dumps(
            {"rationale": "vectorised per instrument", "files": {"factor.py": template.format(name=name)}}
        )


@pytest.fixture(scope="module")
def qlib_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return SyntheticMarket(instruments=30).write_qlib(tmp_path_factory.mktemp("qlib_data"))


def factor_settings(qlib_data: Path, data_dir: Path) -> QlibFactorSettings:
    return QlibFactorSettings.model_validate(
        {
            "qlib": {"provider_uri": str(qlib_data), "region": "cn", "timeout": 600},
            "data_dir": str(data_dir),
            "market": "csi300",
            "benchmark": "SH000300",
            "data_start": "2018-01-01",
            "debug": {"start": "2019-01-01", "end": "2020-12-31", "instruments": 10},
            "segments": {
                "train": ["2018-01-01", "2019-12-31"],
                "valid": ["2020-01-01", "2020-12-31"],
                "test": ["2021-01-01", "2021-12-31"],
            },
            "base_features": {"KLEN": "($high-$low)/$open", "VRATIO5": "Mean($volume, 5)/($volume+1e-12)"},
            "model": {
                "class": "LGBModel",
                "module_path": "qlib.contrib.model.gbdt",
                "kwargs": {"loss": "mse", "learning_rate": 0.05, "num_leaves": 15, "num_threads": 2},
            },
            "backtest": {"topk": 8, "n_drop": 2},
            "promotion": PromotionRule(primary=MetricCriterion(metric="IC")).model_dump(),
            "factor_timeout": 300,
        }
    )


def test_research_run_on_qlib(qlib_data: Path, tmp_path: Path) -> None:
    backend = ScriptedBackend()
    script = ResearchScript()
    script.install(backend)
    llm = LLMClient(backend, LLMSettings(model="scripted", max_retries=0))
    scenario = QlibFactorScenario(factor_settings(qlib_data, tmp_path / "data"))
    settings = AlphaXSettings(
        costeer=CoSTEERSettings(policy="greedy", max_rounds=3, target_score=0.9),
        workflow=WorkflowSettings(run_root=tmp_path / "runs", run_name="e2e", max_loops=2),
    )
    report = AlphaXApplication(settings, llm=llm, scenario=scenario).run()

    meta = json.loads((tmp_path / "data" / "meta.json").read_text())
    assert meta["full_instruments"] == 30 and meta["debug_instruments"] == 10

    trace = report.trace
    assert "IC" in trace.baseline.metrics
    first, second = trace.records
    assert first.feedback.failure is None, first.feedback.reason
    assert first.experiment.result.get("IC") > trace.baseline.get("IC")
    assert first.feedback.decision
    assert "1day.excess_return_with_cost.annualized_return" in first.experiment.result.metrics

    implementation = next(iter(first.experiment.implementations.values()))
    assert implementation.artifacts["included"] == "true"
    values = pd.read_parquet(implementation.artifacts["factor_values"])
    assert list(values.columns) == ["REV5"] and values.index.get_level_values("instrument").nunique() == 30

    coder_prompts = [request.messages[1].content for request in backend.requests_of("factor_coder")]
    assert "Mode: draft" in coder_prompts[0] and "KeyError" in coder_prompts[1]
    assert "Columns keep the $ prefix" in coder_prompts[2]

    assert second.feedback.failure is not None and second.feedback.failure.stage == "run"
    assert "duplicates" in second.feedback.failure.message and "REV5_X2 was dropped" in second.feedback.failure.message
    assert trace.sota().loop == 0

    run_dir = tmp_path / "runs" / "e2e"
    assert (run_dir / "summary.json").exists() and (run_dir / "baseline.json").exists()
    agents = {json.loads(line)["agent"] for line in (run_dir / "llm_calls.jsonl").read_text().splitlines()}
    assert agents == {
        "factor_proposer",
        "factor_designer",
        "factor_coder",
        "factor_critic",
        "reflector",
        "factor_analyst",
    }
    rendered = report.render()
    assert "loop   0  PROMOTED" in rendered and "IC=" in rendered and "failed at run" in rendered
