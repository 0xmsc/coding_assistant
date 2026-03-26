from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from coding_assistant import run_agent
from coding_assistant.agent import AwaitingToolsEvent, AwaitingUserEvent, ContentDeltaEvent, FailedEvent
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent as LLMContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    Tool,
    ToolCall,
    Usage,
    UserMessage,
)


class ScriptedStreamer:
    def __init__(self, script: list[AssistantMessage | Exception]) -> None:
        self.script = list(script)

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        if not self.script:
            raise AssertionError("Streamer script exhausted")

        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action

        if isinstance(action.content, str) and action.content:
            yield LLMContentDeltaEvent(content=action.content)

        yield CompletionEvent(completion=Completion(message=action, usage=Usage(tokens=10, cost=0.0)))


class MockTool(Tool):
    def __init__(self, name: str = "mock_tool") -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Mock external tool"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> str:
        return "tool result"


def make_system_history() -> list[SystemMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


@pytest.mark.asyncio
async def test_run_agent_waits_for_user_without_a_pending_model_turn() -> None:
    events = [event async for event in run_agent(history=make_system_history(), model="test-model", tools=[])]

    assert events == [AwaitingUserEvent()]


@pytest.mark.asyncio
async def test_run_agent_streams_content_then_yields_awaiting_user() -> None:
    events = [
        event
        async for event in run_agent(
            history=[*make_system_history(), UserMessage(content="Hi")],
            model="test-model",
            tools=[],
            streamer=ScriptedStreamer([AssistantMessage(content="Hello from the assistant")]),
        )
    ]

    assert events[0] == ContentDeltaEvent(content="Hello from the assistant")
    assert events[-1] == AwaitingUserEvent(message=AssistantMessage(content="Hello from the assistant"))


@pytest.mark.asyncio
async def test_run_agent_yields_tool_boundary_without_executing_tools() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )

    events = [
        event
        async for event in run_agent(
            history=[*make_system_history(), UserMessage(content="Finish the task")],
            model="test-model",
            tools=[MockTool()],
            streamer=ScriptedStreamer([AssistantMessage(tool_calls=[external_call])]),
        )
    ]

    assert events == [AwaitingToolsEvent(message=AssistantMessage(tool_calls=[external_call]))]


@pytest.mark.asyncio
async def test_run_agent_returns_failed_event_when_streamer_crashes() -> None:
    events = [
        event
        async for event in run_agent(
            history=[*make_system_history(), UserMessage(content="Fail")],
            model="test-model",
            tools=[],
            streamer=ScriptedStreamer([RuntimeError("boom")]),
        )
    ]

    assert events == [FailedEvent(error="boom")]
