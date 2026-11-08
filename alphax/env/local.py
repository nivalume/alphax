"""Run commands as local subprocesses."""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from alphax.core.exceptions import EnvironmentUnavailableError
from alphax.env.base import Environment, EnvResult


class LocalEnvironment(Environment):
    """Subprocesses on this machine, each in its own process group.

    A timed-out command is killed together with every child it started, so a runaway
    factor computation cannot keep the machine busy after the research loop moved on.
    """

    def __init__(
        self,
        python_bin: str | None = None,
        *,
        default_timeout: float | None = 3600.0,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self._python_bin = python_bin or sys.executable
        self._default_timeout = default_timeout
        self._extra_env = dict(extra_env or {})

    @property
    def python_command(self) -> list[str]:
        return [self._python_bin]

    def prepare(self) -> None:
        result = self.run([self._python_bin, "-c", "import sys; print(sys.version)"], Path.cwd(), timeout=60)
        if not result.ok:
            raise EnvironmentUnavailableError(
                f"Python interpreter {self._python_bin!r} is not usable:\n{result.tail()}"
            )

    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> EnvResult:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        limit = timeout if timeout is not None else self._default_timeout
        started = time.monotonic()
        process = subprocess.Popen(
            list(command),
            cwd=workdir,
            env=self._environ(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            raw, _ = process.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_group(process)
            raw, _ = process.communicate()
        output = raw.decode("utf-8", errors="replace")
        if timed_out:
            output += f"\n[alphax] command timed out after {limit} seconds and was killed"
        return EnvResult(
            command=shlex.join(command),
            exit_code=process.returncode if process.returncode is not None else -1,
            output=output,
            seconds=time.monotonic() - started,
            timed_out=timed_out,
        )

    def _environ(self, env: Mapping[str, str] | None) -> dict[str, str]:
        merged = dict(os.environ)
        merged.update(self._extra_env)
        merged.update(env or {})
        merged.setdefault("PYTHONUNBUFFERED", "1")
        return merged

    @staticmethod
    def _kill_group(process: subprocess.Popen[bytes]) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
