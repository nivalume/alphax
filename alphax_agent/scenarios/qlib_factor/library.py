"""The factor library: factors kept by the research so far."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alphax_agent.core.experiment import Implementation
from alphax_agent.core.task import Task
from alphax_agent.core.trace import Trace


@dataclass(frozen=True)
class FactorEntry:
    loop: int
    task: Task
    implementation: Implementation

    @property
    def values_path(self) -> Path | None:
        path = self.implementation.artifacts.get(FactorLibrary.VALUES)
        return Path(path) if path else None


class FactorLibrary:
    """Factors that entered a backtest of an experiment that was promoted.

    The runner marks every implementation it adds to a backtest with ``included`` and the
    path of its full-sample values; the promotion decision lives in the trace. A factor is
    in the library when both hold.
    """

    INCLUDED = "included"
    VALUES = "factor_values"

    def __init__(self, trace: Trace) -> None:
        self._entries = [
            FactorEntry(loop=record.loop, task=task, implementation=implementation)
            for record in trace.accepted()
            if record.experiment is not None
            for task, implementation in record.experiment.accepted()
            if implementation.artifacts.get(self.INCLUDED) == "true"
        ]

    @property
    def entries(self) -> list[FactorEntry]:
        return list(self._entries)

    def names(self) -> set[str]:
        return {entry.task.name for entry in self._entries}

    def render(self) -> str:
        if not self._entries:
            return "The library is empty: only the base features are in the model."
        return "\n\n".join(f"(loop {entry.loop})\n{entry.task.render()}" for entry in self._entries)

    def load_values(self) -> pd.DataFrame | None:
        frames = [pd.read_parquet(path) for entry in self._entries if (path := entry.values_path) and path.exists()]
        if not frames:
            return None
        combined = pd.concat(frames, axis=1)
        return combined.loc[:, ~combined.columns.duplicated(keep="last")]
