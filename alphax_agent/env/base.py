"""Execution environments: where generated code and experiments run."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel

from alphax_agent.core.text import Excerpt


class EnvResult(BaseModel):
    """Outcome of one command."""

    command: str
    exit_code: int
    output: str
    seconds: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def tail(self, max_chars: int = 4000) -> str:
        """The end of the combined output, where tracebacks and summaries are."""
        return Excerpt.tail(self.output, max_chars)


class Environment(ABC):
    """Runs commands in a working directory.

    Implementations decide *where* (local process, container, remote sandbox); the callers
    only see commands, directories and :class:`EnvResult`.
    """

    @abstractmethod
    def prepare(self) -> None:
        """Make sure the environment can run commands; raise if it cannot."""

    @abstractmethod
    def run(
        self,
        command: Sequence[str],
        workdir: Path,
        *,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> EnvResult: ...

    @property
    @abstractmethod
    def python_command(self) -> list[str]:
        """The command prefix that starts the environment's Python interpreter."""

    def run_python(
        self,
        script: str,
        workdir: Path,
        *,
        args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> EnvResult:
        return self.run([*self.python_command, script, *args], workdir, env=env, timeout=timeout)

    @cached_property
    def description(self) -> str:
        """Interpreter and key package versions, shown to the agents that write code."""
        probe = (
            "import platform, importlib.metadata as m\n"
            "print('Python', platform.python_version())\n"
            "for name in ('numpy', 'pandas', 'scipy', 'pyqlib', 'lightgbm'):\n"
            "    try:\n"
            "        print(name, m.version(name))\n"
            "    except m.PackageNotFoundError:\n"
            "        pass\n"
        )
        result = self.run([*self.python_command, "-c", probe], Path.cwd(), timeout=60)
        return result.output.strip() if result.ok else "(runtime description unavailable)"
