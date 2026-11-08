"""Parse model replies into pydantic models."""

from __future__ import annotations

import json
import re
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from alphax.core.exceptions import OutputParsingError

ModelT = TypeVar("ModelT", bound=BaseModel)


class StructuredOutputParser(Generic[ModelT]):
    """Finds the JSON object in a reply and validates it against ``model``.

    Replies are accepted as bare JSON, as a fenced ```json block, or with prose around the
    object, which covers providers that ignore JSON mode.
    """

    _FENCE = re.compile(r"```(?:json|JSON)?\s*\n(.*?)```", re.DOTALL)

    def __init__(self, model: type[ModelT]) -> None:
        self._model = model
        self._decoder = json.JSONDecoder()

    @property
    def model(self) -> type[ModelT]:
        return self._model

    def format_instructions(self) -> str:
        schema = json.dumps(self._model.model_json_schema(), ensure_ascii=False, indent=2)
        return (
            "Reply with a single JSON object, and nothing else, that conforms to this JSON schema:\n"
            f"```json\n{schema}\n```"
        )

    def parse(self, text: str | None) -> ModelT:
        if not text or not text.strip():
            raise OutputParsingError("The reply is empty.")
        payload = self.extract_json(text)
        try:
            return self._model.model_validate(payload)
        except ValidationError as exc:
            raise OutputParsingError(f"The JSON does not match the schema: {exc}") from exc

    def extract_json(self, text: str) -> Any:
        stripped = text.strip()
        candidates = [stripped, *(match.group(1).strip() for match in self._FENCE.finditer(text))]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = self._decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise OutputParsingError("No JSON object was found in the reply.")
