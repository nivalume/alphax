"""Everything a run leaves on disk, in one directory.

::

    <run_dir>/
      settings.json        settings of the run, secrets masked
      run.json             scenario, start/finish time, pid and status (running/finished/failed/interrupted)
      events.jsonl         start, finish or failure of every step of every loop, with its duration
      baseline.json        result of the reference setup
      loops/007.json       one trace record per loop (hypothesis, experiment, feedback)
      llm_calls.jsonl      audit log of every model call: request, response, reasoning, loop/stage/task context
      workspaces/          code, data links and backtest outputs of every attempt
      summary.json         per-loop metrics and decisions, token usage
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from alphax_agent.core.experiment import Experiment, ExperimentResult, Hypothesis
from alphax_agent.core.feedback import ExperimentFeedback
from alphax_agent.core.trace import Trace, TraceRecord
from alphax_agent.llm.types import TokenUsage

RunStatus = Literal["running", "finished", "failed", "interrupted"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ArtifactStore:
    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir).resolve()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self.loops_dir.mkdir(exist_ok=True)
        self._events_lock = threading.Lock()

    @classmethod
    def create(cls, run_root: Path, run_name: str | None = None) -> ArtifactStore:
        name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(Path(run_root) / name)

    @classmethod
    def open(cls, run_dir: Path) -> ArtifactStore:
        if not Path(run_dir).is_dir():
            raise FileNotFoundError(f"Run directory {run_dir} does not exist")
        return cls(run_dir)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def loops_dir(self) -> Path:
        return self._run_dir / "loops"

    @property
    def workspace_dir(self) -> Path:
        return self._run_dir / "workspaces"

    @property
    def llm_audit_path(self) -> Path:
        return self._run_dir / "llm_calls.jsonl"

    @property
    def events_path(self) -> Path:
        return self._run_dir / "events.jsonl"

    def save_settings(self, snapshot: Mapping[str, Any]) -> None:
        self._write_json("settings.json", snapshot)

    def start_run(self, *, scenario: str, options: Mapping[str, Any]) -> None:
        """Mark the run as running in this process. A resumed run keeps its original start time."""
        previous = self.load_run()
        self._write_json(
            "run.json",
            {
                "scenario": scenario,
                "started_at": previous["started_at"] if previous else _now(),
                "finished_at": None,
                "pid": os.getpid(),
                "status": "running",
                "options": dict(options),
                "resumes": previous.get("resumes", 0) + 1 if previous else 0,
            },
        )

    def finish_run(self, status: RunStatus) -> None:
        current = self.load_run()
        if current is not None:
            self._write_json("run.json", {**current, "status": status, "finished_at": _now()})

    def load_run(self) -> dict[str, Any] | None:
        path = self._run_dir / "run.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def append_event(self, **fields: Any) -> None:
        """Add one line to ``events.jsonl``; safe to call from several threads."""
        line = json.dumps({"time": _now(), **fields}, ensure_ascii=False, default=str)
        with self._events_lock, self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def save_baseline(self, result: ExperimentResult | None) -> None:
        if result is not None:
            self._write_json("baseline.json", result.model_dump(mode="json"))

    def load_baseline(self) -> ExperimentResult | None:
        path = self._run_dir / "baseline.json"
        return ExperimentResult.model_validate_json(path.read_text(encoding="utf-8")) if path.exists() else None

    def save_record(self, record: TraceRecord) -> Path:
        path = self.loops_dir / f"{record.loop:03d}.json"
        path.write_text(record.model_dump_json(indent=2, serialize_as_any=True), encoding="utf-8")
        return path

    def load_records(self, experiment_model: type[Experiment] = Experiment) -> list[TraceRecord]:
        """Restore the records of the run, with experiments of the scenario's concrete type."""
        records = []
        for path in sorted(self.loops_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            experiment = data.get("experiment")
            records.append(
                TraceRecord(
                    loop=data["loop"],
                    hypothesis=Hypothesis.model_validate(data["hypothesis"]) if data.get("hypothesis") else None,
                    experiment=experiment_model.model_validate(experiment) if experiment else None,
                    feedback=ExperimentFeedback.model_validate(data["feedback"]),
                )
            )
        return records

    def save_summary(self, trace: Trace, usage: Mapping[str, TokenUsage], headline_metrics: Sequence[str] = ()) -> Path:
        loops = []
        for record in trace.records:
            result = record.experiment.result if record.experiment is not None else None
            loops.append(
                {
                    "loop": record.loop,
                    "hypothesis": record.hypothesis.statement if record.hypothesis else None,
                    "tasks": [task.name for task in record.experiment.tasks] if record.experiment else [],
                    "decision": record.feedback.decision,
                    "failed_stage": record.feedback.failure.stage if record.feedback.failure else None,
                    "metrics": result.metrics if result is not None else {},
                }
            )
        summary = {
            "headline_metrics": list(headline_metrics),
            "baseline": trace.baseline.metrics if trace.baseline is not None else None,
            "loops": loops,
            "accepted_loops": [record.loop for record in trace.accepted()],
            "usage": {agent: value.model_dump() for agent, value in usage.items()},
        }
        return self._write_json("summary.json", summary)

    def _write_json(self, name: str, payload: Any) -> Path:
        # Written to a temporary file and moved into place: a viewer following the run never reads half a file.
        path = self._run_dir / name
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(temporary, path)
        return path
