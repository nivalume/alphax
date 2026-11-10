"""pipeline_version: a hash of the code that judges results. A pass is only valid under the version that produced it."""

from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
# The LLM-facing modules do not judge anything; changing prompts must not invalidate passes.
EXCLUDED = {"llm.py", "pdf.py", "cli.py", "propose.py"}


def pipeline_version(policy_version: str) -> str:
    digest = hashlib.sha256(policy_version.encode())
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name in EXCLUDED:
            continue
        digest.update(str(path.relative_to(PACKAGE)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
