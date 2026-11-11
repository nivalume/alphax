"""End to end on alphax: jfinance export, coding search, look-ahead check, backtest, promotion.

Runs the real alphax engine on a synthetic jfinance store; only the language model is scripted. The interpreter
with alphax is ``$ALPHAX_PYTHON``, or the ``.venv`` of an alphax checkout next to this repository.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from support import ALPHAX_MOMENTUM_STRATEGY, LOOK_AHEAD_STRATEGY, ScriptedBackend, SyntheticJFinance

from alphax_agent.app import AlphaXApplication
from alphax_agent.config.settings import AlphaXSettings, CoSTEERSettings, LLMSettings, WorkflowSettings
from alphax_agent.llm.client import LLMClient
from alphax_agent.scenarios.crypto_strategy import CryptoStrategyScenario, CryptoStrategySettings


def _alphax_python() -> str | None:
    candidates = [
        os.environ.get("ALPHAX_PYTHON"),
        str(Path(__file__).parents[2] / "alphax" / ".venv" / "bin" / "python"),
    ]
    for candidate in filter(None, candidates):
        if Path(candidate).exists():
            probe = subprocess.run([candidate, "-c", "import alphax, pyarrow"], capture_output=True, check=False)
            if probe.returncode == 0:
                return candidate
    return None


PYTHON = _alphax_python()

pytestmark = [
    pytest.mark.alphax,
    pytest.mark.skipif(PYTHON is None, reason="no interpreter with alphax; set ALPHAX_PYTHON"),
]


class ResearchScript:
    """Scripted replies: one hypothesis, a causal momentum rule and a rule that reads the next bar."""

    def install(self, backend: ScriptedBackend) -> None:
        backend.on(
            "strategy_proposer",
            lambda request: json.dumps(
                {"statement": "Slow capital keeps pushing a trend.", "rationale": "Late entrants chase the move."}
            ),
        )
        backend.on(
            "strategy_designer",
            lambda request: json.dumps(
                {
                    "strategies": [
                        {"name": "Momentum", "description": "[Trend Following] Long above the slow mean."},
                        {"name": "Peek", "description": "[Trend Following] Follows the next bar."},
                    ]
                }
            ),
        )
        backend.on("strategy_coder", self.code)
        backend.on(
            "strategy_critic",
            lambda request: json.dumps({"consistent": True, "score": 0.9, "critique": "Implements the rule."}),
        )
        backend.on("reflector", lambda request: json.dumps({"lessons": [], "reinforced": []}))
        backend.on(
            "strategy_analyst",
            lambda request: json.dumps(
                {
                    "observations": "The book trended.",
                    "hypothesis_evaluation": "Consistent with slow capital.",
                    "hypothesis_supported": True,
                    "next_hypothesis": "Gate the trend on volatility.",
                    "reason": "See metrics.",
                }
            ),
        )

    @staticmethod
    def code(request) -> str:
        task = request.messages[1].content
        name = re.search(r"Task: (\w+)", task) or re.search(r"\b(Momentum|Peek)\b", task)
        source = LOOK_AHEAD_STRATEGY if name and name.group(1) == "Peek" else ALPHAX_MOMENTUM_STRATEGY
        return json.dumps({"rationale": "vectorised", "files": {"strategy.py": source}})


def test_research_run_on_alphax(tmp_path: Path) -> None:
    market = SyntheticJFinance(instruments=6)
    store = market.write(tmp_path / "jfinance")
    settings = CryptoStrategySettings.model_validate(
        {
            "alphax": {"python_bin": PYTHON, "data_dir": str(store), "timeout": 600},
            "period": {"start": "2021-01-01", "oos_start": "2023-01-01", "end": "2024-01-01"},
            "universe": {"max_instruments": 5, "min_bars": 200},
            "debug": {"instruments": 3, "bars": 300},
            "backtest": {"warmup_bars": 60},
            "cache_dir": str(tmp_path / "cache"),
        }
    )
    backend = ScriptedBackend()
    ResearchScript().install(backend)
    llm = LLMClient(backend, LLMSettings(model="scripted", max_retries=0))
    platform = AlphaXSettings(
        costeer=CoSTEERSettings(policy="greedy", max_rounds=2, target_score=0.9),
        workflow=WorkflowSettings(run_root=tmp_path / "runs", run_name="e2e", max_loops=1),
    )
    report = AlphaXApplication(platform, llm=llm, scenario=CryptoStrategyScenario(settings)).run()

    meta = json.loads((tmp_path / "cache" / "meta.json").read_text())
    universe = [instrument["symbol"] for instrument in meta["universe"]["instruments"]]
    assert len(universe) == 5 and meta["full"]["warmup_bars"] == 60 and len(meta["debug"]["symbols"]) == 3
    # The most liquid instruments by median in-sample traded value, nothing after oos_start.
    in_sample = (market.days >= np.datetime64("2021-01-01")) & (market.days < np.datetime64("2023-01-01"))
    traded = np.nanmedian(market.bars["quote_volume"][in_sample], axis=0)
    assert universe == [market.symbols[column] for column in np.argsort(-traded)][:5]

    (record,) = report.trace.records
    assert record.feedback.failure is None, record.feedback.reason
    experiment = record.experiment
    implementations = {experiment.task(task_id).name: impl for task_id, impl in experiment.implementations.items()}
    assert implementations["Momentum"].feedback.accepted
    assert not implementations["Peek"].feedback.accepted
    assert "change when the bars after it are removed" in implementations["Peek"].feedback.render()

    result = experiment.result
    assert result.artifacts["selected_strategy"] == "Momentum"
    assert {"is.portfolio.sharpe", "is.portfolio.max_drawdown", "is.trade_count", "is.benchmark.sharpe"} <= set(
        result.metrics
    )
    assert result.get("is.trade_count") > 0 and -1.0 <= result.get("is.portfolio.max_drawdown") <= 0.0
    assert not any(name.startswith("oos.") for name in result.metrics)
    held_out = json.loads(Path(result.artifacts["oos_metrics"]).read_text())
    assert "portfolio.sharpe" in held_out and held_out["bars"] == 365
    assert record.feedback.decision  # the first result is the first incumbent

    analyst = backend.requests_of("strategy_analyst")[0].messages[1].content
    assert "is.portfolio.sharpe" in analyst and "oos" not in analyst
    coder = backend.requests_of("strategy_coder")[0].messages[0].content
    assert "alphax-trader 0.3" in coder or "alphax-trader" in coder
    assert "loop   0  PROMOTED" in report.render() and "is.portfolio.sharpe=" in report.render()
