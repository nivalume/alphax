from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel
from support import ScriptedBackend

from alphax.agents.agent import Agent
from alphax.agents.tool import Tool
from alphax.core.exceptions import AgentError, OutputParsingError
from alphax.core.prompts import PromptLibrary


class Verdict(BaseModel):
    label: str
    confidence: float


class EchoArgs(BaseModel):
    text: str


class EchoTool(Tool[EchoArgs]):
    name = "echo"
    description = "Echo the text in upper case"
    args_model = EchoArgs

    def run(self, args: EchoArgs) -> str:
        return args.text.upper()


PROMPTS = PromptLibrary({"judge.system": "You judge {{ topic }}.", "judge.user": "Judge: {{ item }}"})


class JudgeAgent(Agent[Verdict]):
    name = "judge"
    output_model = Verdict
    system_template = "judge.system"
    user_template = "judge.user"
    labels: ClassVar[set[str]] = {"good", "bad"}

    def judge(self, item: str) -> Verdict:
        return self._ask(topic="items", item=item)

    def _validate(self, output: Verdict) -> Verdict:
        if output.label not in self.labels:
            raise OutputParsingError(f"label must be one of {sorted(self.labels)}")
        return output


def test_prompt_rendering_and_schema(backend: ScriptedBackend, llm) -> None:
    backend.on("judge", ['{"label": "good", "confidence": 0.9}'])
    verdict = JudgeAgent(llm, PROMPTS).judge("apple")
    assert verdict == Verdict(label="good", confidence=0.9)
    request = backend.requests_of("judge")[0]
    assert request.messages[0].content.startswith("You judge items.")
    assert '"confidence"' in request.messages[0].content
    assert request.messages[1].content == "Judge: apple"
    assert request.json_mode and request.tools is None


def test_tool_loop(backend: ScriptedBackend, llm) -> None:
    backend.on("judge", [ScriptedBackend.tool_call("echo", {"text": "hi"}), '{"label": "bad", "confidence": 0.1}'])
    agent = JudgeAgent(llm, PROMPTS, tools=[EchoTool()])
    assert agent.judge("pear").label == "bad"
    second = backend.requests_of("judge")[1]
    assert second.tools and second.tools[0]["function"]["name"] == "echo"
    assert second.messages[-1].role == "tool" and second.messages[-1].content == "HI"


def test_tool_errors_are_returned_to_the_model(backend: ScriptedBackend, llm) -> None:
    backend.on(
        "judge",
        [
            ScriptedBackend.tool_call("missing", {}),
            ScriptedBackend.tool_call("echo", {"wrong": 1}),
            '{"label": "good", "confidence": 1}',
        ],
    )
    JudgeAgent(llm, PROMPTS, tools=[EchoTool()]).judge("x")
    third = backend.requests_of("judge")[2]
    tool_messages = [message.content for message in third.messages if message.role == "tool"]
    assert "unknown tool" in tool_messages[0]
    assert "invalid arguments" in tool_messages[1]


def test_tool_budget(backend: ScriptedBackend, llm) -> None:
    calls = [ScriptedBackend.tool_call("echo", {"text": str(i)}, call_id=f"c{i}") for i in range(3)]
    backend.on("judge", [*calls, '{"label": "good", "confidence": 1}'])
    JudgeAgent(llm, PROMPTS, tools=[EchoTool()], max_tool_rounds=2).judge("x")
    last = backend.requests_of("judge")[-1]
    assert "tool budget" in last.messages[-1].content


def test_repairs_invalid_output(backend: ScriptedBackend, llm) -> None:
    backend.on("judge", ["not json", '{"label": "ugly", "confidence": 1}', '{"label": "good", "confidence": 1}'])
    assert JudgeAgent(llm, PROMPTS).judge("x").label == "good"
    retry = backend.requests_of("judge")[2]
    assert "label must be one of" in retry.messages[-1].content


def test_gives_up_after_parse_budget(backend: ScriptedBackend, llm) -> None:
    backend.on("judge", ["no", "no", "no"])
    with pytest.raises(AgentError, match="no valid output"):
        JudgeAgent(llm, PROMPTS, max_parse_retries=2).judge("x")
