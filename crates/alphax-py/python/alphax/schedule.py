"""Run a job on a schedule and report the outcome to notifiers.

The library never sends anything over the network itself. A notifier is any
callable taking ``(subject, text)``; :class:`Command` runs a shell command
with the message on its input, :class:`File` appends it to a file, and
``examples/notifiers.py`` has Telegram and Slack ones built on that idea::

    from alphax import schedule

    job = schedule.Job(lambda: alphax.config.run("daily.toml"), name="daily",
                       notify=[schedule.File("runs.log"), schedule.Command("mail -s alphax me@example.com")])
    schedule.every(job, at="06:30")          # blocks, runs it daily at 06:30 local time
    schedule.crontab("daily.toml", at="06:30")   # or: a crontab line for `alphax run`
"""

from __future__ import annotations

import datetime as dt
import shlex
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Command", "File", "Job", "crontab", "every", "next_time"]

Notifier = Callable[[str, str], Any]


@dataclass
class Command:
    """Runs ``command`` with the message (subject, blank line, text) on
    standard input."""

    command: str

    def __call__(self, subject: str, text: str) -> None:
        subprocess.run(shlex.split(self.command), input=f"{subject}\n\n{text}", text=True, check=True)


@dataclass
class File:
    """Appends the message to ``path`` with a UTC time stamp."""

    path: str

    def __call__(self, subject: str, text: str) -> None:
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {subject}\n{text}\n\n")


def _summary(result: Any) -> str:
    if isinstance(result, dict) and "stats" in result:
        lines = []
        for params, per_account in zip(result.get("params", []), result["stats"]):
            for a, st in enumerate(per_account):
                tr, sr = st.get("Total Return [%]"), st.get("Sharpe Ratio")
                lines.append(f"{params or 'run'} account {a}: total return {tr}%, Sharpe {sr}")
        return "\n".join(lines)
    return repr(result)


@dataclass
class Job:
    """A callable with a name and notifiers: each run reports success with a
    summary of what it returned, or failure with the traceback."""

    func: Callable[[], Any]
    name: str = "alphax job"
    notify: Iterable[Notifier] = field(default_factory=list)
    summarize: Callable[[Any], str] = _summary

    def __call__(self) -> Any:
        try:
            result = self.func()
        except Exception:
            self._send(f"{self.name} failed", traceback.format_exc())
            raise
        self._send(f"{self.name} done", self.summarize(result))
        return result

    def _send(self, subject: str, text: str) -> None:
        for n in self.notify:
            try:
                n(subject, text)
            except Exception as e:  # a notifier failing must not hide the job's outcome
                print(f"notifier {n!r} failed: {e}", file=sys.stderr)


def next_time(now: dt.datetime, *, at: str | None = None, interval: float | None = None) -> dt.datetime:
    """The next run time after ``now``: daily at ``"HH:MM"``, or ``now +
    interval`` seconds."""
    if (at is None) == (interval is None):
        raise ValueError("give exactly one of at and interval")
    if interval is not None:
        return now + dt.timedelta(seconds=interval)
    hour, minute = (int(x) for x in at.split(":"))
    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return candidate if candidate > now else candidate + dt.timedelta(days=1)


def every(job: Callable[[], Any], *, at: str | None = None, interval: float | None = None,
          times: int | None = None, sleep: Callable[[float], Any] = time.sleep,
          clock: Callable[[], dt.datetime] = dt.datetime.now) -> None:
    """Runs ``job`` daily ``at`` a local time or every ``interval`` seconds,
    ``times`` times (forever by default). A failing run is reported by the
    job's notifiers and the schedule goes on."""
    done = 0
    while times is None or done < times:
        wait = (next_time(clock(), at=at, interval=interval) - clock()).total_seconds()
        sleep(max(0.0, wait))
        try:
            job()
        except Exception:
            pass
        done += 1


def crontab(config: str | Path, *, at: str = "06:30", weekdays: str = "*", python: str | None = None) -> str:
    """A crontab line that runs ``alphax run <config>`` daily ``at``
    (``weekdays`` in cron's day-of-week field, e.g. ``"1-5"``)."""
    hour, minute = (int(x) for x in at.split(":"))
    exe = shlex.quote(python or sys.executable)
    cfg = shlex.quote(str(Path(config).resolve()))
    return f"{minute} {hour} * * {weekdays} {exe} -m alphax.cli run {cfg}"
