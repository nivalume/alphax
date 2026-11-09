"""On-disk response cache keyed by the full request."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from alphax_agent.llm.types import ChatResponse


class ResponseCache:
    """One JSON file per request hash.

    Replaying a run against the cache reproduces it exactly without calling the provider,
    which is how a research result is audited after the fact.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory).expanduser()
        self._directory.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> ChatResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        response = ChatResponse.model_validate_json(path.read_text(encoding="utf-8"))
        return response.model_copy(update={"cached": True})

    def put(self, key: str, response: ChatResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(response.model_dump_json())
        os.replace(temp, path)

    def _path(self, key: str) -> Path:
        return self._directory / key[:2] / f"{key}.json"
