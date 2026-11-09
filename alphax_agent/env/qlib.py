"""The qlib runtime: data provider, workflow runner and result export."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path

from jinja2 import Environment as JinjaEnvironment
from jinja2 import StrictUndefined
from pydantic import BaseModel, Field

from alphax_agent.core.exceptions import EnvironmentUnavailableError, ExecutionError
from alphax_agent.env.base import Environment, EnvResult
from alphax_agent.env.local import LocalEnvironment


class QlibEnvSettings(BaseModel):
    provider_uri: str = Field(default="~/.qlib/qlib_data/cn_data", description="qlib binary data directory")
    region: str = "cn"
    python_bin: str | None = Field(default=None, description="Interpreter with qlib installed; default: this one")
    timeout: float = Field(default=3600.0, description="Limit of one qlib workflow run, seconds")

    @property
    def provider_path(self) -> Path:
        return Path(self.provider_uri).expanduser().resolve()


class QlibWorkflowResult(BaseModel):
    metrics: dict[str, float]
    report_path: str | None = None
    log: str = ""


class QlibEnvironment(Environment):
    """Qlib on top of an execution backend.

    The backend decides where commands run (a local interpreter by default; a container or
    remote sandbox can be passed instead); this class knows how to check a qlib
    installation, run a workflow config and read its recorder.
    """

    METRICS_FILE = "qlib_metrics.json"
    REPORT_FILE = "portfolio_report.parquet"
    READER_SCRIPT = "alphax_read_results.py"
    URI_FOLDER = "mlruns"

    def __init__(self, settings: QlibEnvSettings, backend: Environment | None = None) -> None:
        self._settings = settings
        self._backend = backend or LocalEnvironment(settings.python_bin, default_timeout=settings.timeout)
        self._templates = JinjaEnvironment(undefined=StrictUndefined, keep_trailing_newline=True)

    @property
    def settings(self) -> QlibEnvSettings:
        return self._settings

    @property
    def python_command(self) -> list[str]:
        return self._backend.python_command

    def prepare(self) -> None:
        self._backend.prepare()
        probe = self._backend.run(
            [*self.python_command, "-c", "import qlib; print(qlib.__file__)"], Path.cwd(), timeout=120
        )
        if not probe.ok:
            raise EnvironmentUnavailableError(f"qlib is not importable in the execution environment:\n{probe.tail()}")
        if not self._settings.provider_path.exists():
            raise EnvironmentUnavailableError(f"qlib data directory {self._settings.provider_path} does not exist")

    def trading_calendar(self, freq: str = "day") -> list[str]:
        """Trading dates of the data provider, ``YYYY-MM-DD``, oldest first."""
        path = self._settings.provider_path / "calendars" / f"{freq}.txt"
        if not path.exists():
            raise EnvironmentUnavailableError(f"qlib calendar {path} does not exist")
        return [line.strip()[:10] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> EnvResult:
        return self._backend.run(command, workdir, env=env, timeout=timeout)

    def run_workflow(
        self,
        workdir: Path,
        config_name: str,
        *,
        experiment_name: str = "alphax",
        env: Mapping[str, str] | None = None,
    ) -> QlibWorkflowResult:
        """Run ``qrun`` on a config in ``workdir`` and export the recorder's metrics."""
        workdir = Path(workdir)
        command = [
            *self.python_command,
            "-m",
            "qlib.cli.run",
            config_name,
            f"--experiment_name={experiment_name}",
            f"--uri_folder={self.URI_FOLDER}",
        ]
        result = self.run(command, workdir, env=env, timeout=self._settings.timeout)
        if not result.ok:
            raise ExecutionError(f"qlib workflow {config_name} failed (exit {result.exit_code}):\n{result.tail()}")
        (workdir / self.READER_SCRIPT).write_text(self._reader_script(experiment_name), encoding="utf-8")
        export = self.run_python(self.READER_SCRIPT, workdir, env=env, timeout=600)
        metrics_path = workdir / self.METRICS_FILE
        if not export.ok or not metrics_path.exists():
            raise ExecutionError(f"Reading the qlib recorder failed:\n{export.tail()}")
        report = workdir / self.REPORT_FILE
        return QlibWorkflowResult(
            metrics=json.loads(metrics_path.read_text(encoding="utf-8")),
            report_path=str(report) if report.exists() else None,
            log=result.tail(6000),
        )

    def _reader_script(self, experiment_name: str) -> str:
        source = resources.files("alphax_agent.env").joinpath("qlib_read_results.py.j2").read_text(encoding="utf-8")
        return self._templates.from_string(source).render(
            provider_uri=str(self._settings.provider_path),
            region=self._settings.region,
            uri_folder=self.URI_FOLDER,
            experiment_name=experiment_name,
            metrics_file=self.METRICS_FILE,
            report_file=self.REPORT_FILE,
        )
