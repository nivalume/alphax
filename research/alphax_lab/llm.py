"""Minimal OpenAI-compatible chat client returning JSON.

Configuration: ALPHAX_LAB_MODEL / ALPHAX_LAB_API_BASE / ALPHAX_LAB_API_KEY, falling back to
CHAT_MODEL / OPENAI_API_BASE / OPENAI_API_KEY (the RD-Agent .env names). Values come from
the environment, then research/.env, then ../RD-Agent/.env; ALPHAX_LAB_ENV_FILE overrides
which file is read.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

PACKAGE_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
RDAGENT_ENV_FILE = Path(__file__).resolve().parents[3] / "RD-Agent" / ".env"


def _read(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$", line)
            if match and not line.lstrip().startswith("#"):
                values[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return values


def _env_file() -> dict[str, str]:
    override = os.environ.get("ALPHAX_LAB_ENV_FILE")
    paths = [Path(override).expanduser()] if override else [PACKAGE_ENV_FILE, RDAGENT_ENV_FILE]
    values: dict[str, str] = {}
    for path in reversed(paths):  # earlier paths win
        values.update(_read(path))
    return values


def _setting(primary: str, fallback: str, file: dict[str, str]) -> str | None:
    return os.environ.get(primary) or file.get(primary) or os.environ.get(fallback) or file.get(fallback)


class LLM:
    """Every prompt and reply is kept in `transcript` so what the model saw can be audited."""

    def __init__(self) -> None:
        file = _env_file()
        model = _setting("ALPHAX_LAB_MODEL", "CHAT_MODEL", file)
        if not model:
            raise RuntimeError("set ALPHAX_LAB_MODEL (or CHAT_MODEL) to use the LLM")
        self.model = model.split("/", 1)[1] if model.startswith("openai/") else model
        self.base_url = _setting("ALPHAX_LAB_API_BASE", "OPENAI_API_BASE", file)
        self.api_key = _setting("ALPHAX_LAB_API_KEY", "OPENAI_API_KEY", file)
        self.transcript: list[dict] = []

    def chat_json(self, system: str, user: str, attempts: int = 3) -> dict:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=600)
        error: Exception | None = None
        for _ in range(attempts):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.5,
                )
                text = response.choices[0].message.content or ""
                self.transcript.append({"system": system, "user": user, "reply": text})
                return parse_json(text)
            except (ValueError, OSError) as exc:
                error = exc
                time.sleep(2)
            except Exception as exc:  # openai raises its own hierarchy for transport errors
                error = exc
                time.sleep(5)
        raise RuntimeError(f"LLM call failed after {attempts} attempts: {error}")


def parse_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("reply contains no JSON object")
    return json.loads(body[start : end + 1])
