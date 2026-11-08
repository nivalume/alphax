from __future__ import annotations

import json

import pytest
from pydantic import BaseModel
from support import ScriptedBackend

from alphax.config.settings import LLMSettings
from alphax.core.exceptions import LLMError, LLMTransientError, OutputParsingError
from alphax.llm.backend import ChatBackend, LiteLLMBackend
from alphax.llm.cache import ResponseCache
from alphax.llm.client import LLMAuditLog, LLMClient
from alphax.llm.parser import StructuredOutputParser
from alphax.llm.types import ChatMessage, ChatRequest, ChatResponse, TokenUsage


class Answer(BaseModel):
    value: int
    note: str = ""


class FlakyBackend(ChatBackend):
    def __init__(self, failures: int, error: type[Exception] = LLMTransientError) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    @property
    def model(self) -> str:
        return "flaky"

    def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("boom")
        return ChatResponse(content='{"value": 1}', usage=TokenUsage(calls=1, prompt_tokens=3, completion_tokens=2))


class TestStructuredOutputParser:
    parser = StructuredOutputParser(Answer)

    @pytest.mark.parametrize(
        "text",
        [
            '{"value": 3}',
            'Here you go:\n```json\n{"value": 3, "note": "x"}\n```',
            'The answer is {"value": 3} as requested.',
            '```\n{"value": 3}\n```',
        ],
    )
    def test_accepts_common_reply_shapes(self, text: str) -> None:
        assert self.parser.parse(text).value == 3

    def test_rejects_missing_json(self) -> None:
        with pytest.raises(OutputParsingError, match="No JSON object"):
            self.parser.parse("no json here")

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(OutputParsingError, match="schema"):
            self.parser.parse('{"value": "not a number"}')

    def test_format_instructions_contain_schema(self) -> None:
        assert '"value"' in self.parser.format_instructions()


class TestLLMClient:
    def test_retries_transient_errors(self) -> None:
        backend = FlakyBackend(failures=2)
        delays: list[float] = []
        client = LLMClient(backend, LLMSettings(model="flaky", max_retries=3, retry_backoff=2.0), sleeper=delays.append)
        response = client.chat([ChatMessage.user("hi")], agent="tester")
        assert response.content == '{"value": 1}'
        assert backend.calls == 3
        assert delays == [1.0, 2.0]

    def test_gives_up_after_budget(self) -> None:
        client = LLMClient(FlakyBackend(failures=5), LLMSettings(model="flaky", max_retries=1), sleeper=lambda _: None)
        with pytest.raises(LLMTransientError):
            client.chat([ChatMessage.user("hi")])

    def test_does_not_retry_permanent_errors(self) -> None:
        backend = FlakyBackend(failures=1, error=LLMError)
        client = LLMClient(backend, LLMSettings(model="flaky", max_retries=3), sleeper=lambda _: None)
        with pytest.raises(LLMError):
            client.chat([ChatMessage.user("hi")])
        assert backend.calls == 1

    def test_usage_cache_and_audit(self, tmp_path) -> None:
        backend = ScriptedBackend({"a": ['{"value": 1}'], "b": ['{"value": 2}']})
        audit = LLMAuditLog(tmp_path / "calls.jsonl")
        client = LLMClient(backend, LLMSettings(model="scripted"), cache=ResponseCache(tmp_path / "cache"), audit=audit)
        first = client.chat([ChatMessage.user("same")], agent="a")
        second = client.chat([ChatMessage.user("same")], agent="a")
        client.chat([ChatMessage.user("other")], agent="b")
        assert not first.cached and second.cached
        assert len(backend.requests) == 2
        assert client.usage.calls == 2
        assert set(client.usage_by_agent()) == {"a", "b"}
        lines = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
        assert [line["agent"] for line in lines] == ["a", "a", "b"]
        assert lines[0]["request"]["messages"][0]["content"] == "same"


class TestLiteLLMBackend:
    def test_mock_response_round_trip(self) -> None:
        settings = LLMSettings(model="gpt-4o-mini", extra_params={"mock_response": '{"value": 7}'})
        response = LiteLLMBackend(settings).complete(ChatRequest(messages=[ChatMessage.user("hi")], json_mode=True))
        assert StructuredOutputParser(Answer).parse(response.content).value == 7
        assert response.usage.calls == 1

    def test_tool_message_format(self) -> None:
        call = ScriptedBackend.tool_call("profile_data", {"column": "$close"}).tool_calls[0]
        assistant = ChatMessage.assistant(None, [call]).to_provider()
        assert assistant["tool_calls"][0]["function"]["name"] == "profile_data"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"column": "$close"}
        tool = ChatMessage.tool(call, "result").to_provider()
        assert tool == {"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "profile_data"}
