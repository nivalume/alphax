"""Append-only research ledger in its own git repository.

The ledger counts every test that looked at real data. Gate thresholds are
functions of those counts, so the count must not be resettable: entries are
hash-chained, each append is committed, and nothing is ever rewritten.

Entry kinds:
  legacy      trials that happened before the ledger (counts only)
  registered  a frozen spec + card (hash = spec id)
  run         one in-sample evaluation of a registered spec (an IS test)
  control     a calibration run (counted as an IS test when it used real data)
  oos_batch   one out-of-sample batch (every spec in it is an OOS test)
  forward     a forward shadow-tracking checkpoint
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

GENESIS = "0" * 64


class LedgerError(RuntimeError):
    pass


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Ledger:
    def __init__(self, root: str | Path, *, commit: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.file = self.root / "ledger.jsonl"
        self.commit_enabled = commit
        self.root.mkdir(parents=True, exist_ok=True)
        for sub in ("pending", "registry", "reports"):
            (self.root / sub).mkdir(exist_ok=True)
        if commit and not (self.root / ".git").exists():
            self._git("init", "-q")
        self.file.touch()
        self.entries = self._read()

    # ------------------------------------------------------------ storage

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        result = subprocess.run(["git", *args], cwd=self.root, capture_output=True, text=True)
        if result.returncode and args[0] != "diff":
            raise LedgerError(f"git {' '.join(args)} failed in {self.root}: {result.stderr.strip() or result.stdout.strip()}")
        return result

    def _read(self) -> list[dict]:
        entries, previous = [], GENESIS
        for number, line in enumerate(self.file.read_text().splitlines(), 1):
            entry = json.loads(line)
            if entry.get("prev_hash") != previous or entry.get("hash") != _digest(entry):
                raise LedgerError(f"ledger chain broken at line {number}; the ledger was edited")
            entries.append(entry)
            previous = entry["hash"]
        return entries

    def append(self, kind: str, payload: dict, message: str | None = None, files: Sequence[Path] = ()) -> dict:
        entry = {
            "seq": len(self.entries),
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "prev_hash": self.entries[-1]["hash"] if self.entries else GENESIS,
            **payload,
        }
        entry["hash"] = _digest(entry)
        with self.file.open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        self.entries.append(entry)
        self.commit(message or f"{kind} {payload.get('spec_id', '')}".strip(), [self.file, *files])
        return entry

    def commit(self, message: str, files: Sequence[Path]) -> None:
        if not self.commit_enabled:
            return
        self._git("add", *[str(Path(f).relative_to(self.root)) for f in files])
        if not self._git("diff", "--cached", "--quiet").returncode:
            return  # nothing changed, e.g. re-running `report` without new entries
        self._git("-c", "user.name=alphax-lab", "-c", "user.email=alphax-lab@localhost", "commit", "-q", "-m", message)

    # ------------------------------------------------------------ counts

    def in_sample_tests(self) -> int:
        """Every look at in-sample data: runs, real-data controls, and imported legacy trials."""
        total = 0
        for e in self.entries:
            if e["kind"] == "run" or (e["kind"] == "control" and e.get("real_data")):
                total += 1
            elif e["kind"] == "legacy":
                total += int(e["tests"])
        return total

    def oos_tests(self) -> int:
        batches = sum(len(e["spec_ids"]) for e in self.entries if e["kind"] == "oos_batch")
        return batches + sum(int(e.get("oos_tests", 0)) for e in self.entries if e["kind"] == "legacy")

    def oos_batches(self) -> list[dict]:
        return [e for e in self.entries if e["kind"] == "oos_batch"]

    def oos_batches_used(self) -> int:
        """Batches, plus one for legacy runs that already read the out-of-sample period."""
        legacy = any(int(e.get("oos_tests", 0)) for e in self.entries if e["kind"] == "legacy")
        return len(self.oos_batches()) + int(legacy)

    # ------------------------------------------------------------ lookups

    def registered(self) -> dict[str, dict]:
        return {e["spec_id"]: e for e in self.entries if e["kind"] == "registered"}

    def runs(self) -> dict[str, dict]:
        return {e["spec_id"]: e for e in self.entries if e["kind"] == "run"}

    def oos_results(self) -> dict[str, dict]:
        out = {}
        for batch in self.oos_batches():
            out.update(batch["results"])
        return out

    def forward_checkpoints(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for e in self.entries:
            if e["kind"] == "forward":
                out.setdefault(e["spec_id"], []).append(e)
        return out

    def tried_structures(self) -> dict[str, str]:
        """Titles and templates of everything tested so far, for de-duplication. No results."""
        tried = {}
        for e in self.entries:
            if e["kind"] == "legacy":
                tried.update({f"legacy: {name}": expr for name, expr in (e.get("structures") or {}).items()})
        tried.update({e["card"]["title"]: e["spec"]["expression"] for e in self.registered().values()})
        return tried
