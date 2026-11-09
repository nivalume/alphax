"""Composition root: settings → LLM client → scenario → research workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from alphax.config.settings import AlphaXSettings
from alphax.core.experiment import ExperimentResult
from alphax.core.registry import ScenarioRegistry, scenario_registry
from alphax.core.scenario import Scenario
from alphax.core.trace import Trace
from alphax.llm.client import LLMAuditLog, LLMClient
from alphax.llm.types import TokenUsage
from alphax.workflow.artifacts import ArtifactStore
from alphax.workflow.graph import ResearchWorkflow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunReport:
    run_dir: Path
    trace: Trace
    usage: dict[str, TokenUsage]
    headline_metrics: tuple[str, ...] = ()

    def render(self) -> str:
        lines = [f"Run directory: {self.run_dir}"]
        if self.trace.baseline is not None:
            lines.append(f"baseline  {self._metrics(self.trace.baseline)}")
        for record in self.trace.records:
            if record.feedback.failure is not None:
                status = f"failed at {record.feedback.failure.stage}"
            else:
                status = "PROMOTED" if record.feedback.decision else "not promoted"
            result = record.experiment.result if record.experiment is not None else None
            statement = record.hypothesis.statement if record.hypothesis else "(no hypothesis)"
            lines.append(f"loop {record.loop:3d}  {status:<20} {self._metrics(result)}  {statement[:80]}")
        total = sum(self.usage.values(), TokenUsage())
        tokens = f"{total.prompt_tokens}+{total.completion_tokens}"
        lines.append(f"LLM calls: {total.calls}, tokens: {tokens}, cost: ${total.cost:.4f}")
        return "\n".join(lines)

    def _metrics(self, result: ExperimentResult | None) -> str:
        if result is None:
            return "(no result)"
        values = [f"{name}={result.get(name):.4f}" for name in self.headline_metrics if result.get(name) is not None]
        return ", ".join(values) or "(no headline metric)"


class AlphaXApplication:
    def __init__(
        self,
        settings: AlphaXSettings,
        *,
        llm: LLMClient | None = None,
        scenario: Scenario | None = None,
        registry: ScenarioRegistry = scenario_registry,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._scenario = scenario
        self._registry = registry

    @property
    def settings(self) -> AlphaXSettings:
        return self._settings

    @cached_property
    def scenario(self) -> Scenario:
        if self._scenario is not None:
            return self._scenario
        selection = self._settings.scenario
        return self._registry.get(selection.name).from_options(selection.options)

    def prepare(self, *, force: bool = False) -> None:
        self.scenario.prepare(force=force)

    def run(self, *, loops: int | None = None, resume: Path | None = None, until: str | None = None) -> RunReport:
        settings = self._settings
        store = (
            ArtifactStore.open(resume)
            if resume is not None
            else ArtifactStore.create(settings.workflow.run_root, settings.workflow.run_name)
        )
        store.save_settings(settings.snapshot())
        llm = self._client(store)
        scenario = self.scenario
        scenario.prepare()
        components = scenario.assemble(llm, settings, store.workspace_dir)
        workflow = ResearchWorkflow(components, settings.workflow, store)
        history = store.load_records(scenario.experiment_model) if resume is not None else []
        baseline = store.load_baseline() if resume is not None else None
        logger.info(
            "Run %s: scenario %s, %d loop(s)%s",
            store.run_dir.name,
            scenario.name,
            loops or settings.workflow.max_loops,
            f", resuming after {len(history)}" if history else "",
        )
        try:
            trace = workflow.run(loops=loops, history=history, baseline=baseline, until=until)
        finally:
            usage = llm.usage_by_agent()
        store.save_summary(trace, usage, scenario.headline_metrics)
        return RunReport(run_dir=store.run_dir, trace=trace, usage=usage, headline_metrics=scenario.headline_metrics)

    def _client(self, store: ArtifactStore) -> LLMClient:
        if self._llm is None:
            return LLMClient.from_settings(self._settings.llm, audit_path=store.llm_audit_path)
        self._llm.attach_audit(LLMAuditLog(store.llm_audit_path))
        return self._llm
