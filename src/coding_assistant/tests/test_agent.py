from __future__ import annotations

import json
from typing import Any

import pytest

from coding_assistant import run_agent
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    FunctionCall,
    SystemMessage,
    Tool,
    ToolCall,
    Usage,
    UserMessage,
)
from coding_assistant.tool_policy import ToolPolicy


class ScriptedCompleter:
    def __init__(self, script: list[AssistantMessage | Exception]) -> None:
        self.script = list(script)

    async def __call__(self, messages: Any, *, model: Any, tools: Any, callbacks: Any) -> Completion:
        if not self.script:
            raise AssertionError("Completer script exhausted")

        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action

        if isinstance(action.content, str) and action.content:
            callbacks.on_content_chunk(action.content)
        callbacks.on_chunks_end()
        return Completion(message=action, usage=Usage(tokens=10, cost=0.0))


class MockTool(Tool):
    def __init__(self, name: str = "mock_tool", result: str = "tool result") -> None:
        self._name = name
        self._result = result
        self.calls: list[dict[str, Any]] = []

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Mock external tool"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> str:
        self.calls.append(parameters)
        return self._result


class DenyPolicy(ToolPolicy):
    async def before_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        return "Tool execution denied."


def make_system_history() -> list[SystemMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


@pytest.mark.asyncio
async def test_run_agent_waits_for_user_without_a_pending_model_turn() -> None:
    result = await run_agent(
        history=make_system_history(),
        model="test-model",
        tools=[],
    )

    assert result.status == "awaiting_user"
    assert result.history == make_system_history()


@pytest.mark.asyncio
async def test_run_agent_streams_and_returns_updated_history() -> None:
    deltas: list[str] = []
    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Hi")],
        model="test-model",
        tools=[],
        completer=ScriptedCompleter([AssistantMessage(content="Hello from the assistant")]),
        on_delta=deltas.append,
    )

    assert result.status == "awaiting_user"
    assert deltas == ["Hello from the assistant", "\n"]
    assert [message.role for message in result.history] == ["system", "user", "assistant"]


@pytest.mark.asyncio
async def test_run_agent_executes_tool_requests() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments=json.dumps({})),
    )
    tool = MockTool()

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Finish the task")],
        model="test-model",
        tools=[tool],
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[external_call]),
                AssistantMessage(content="done"),
            ]
        ),
    )

    assert result.status == "awaiting_user"
    assert tool.calls == [{}]
    tool_messages = [message for message in result.history if message.role == "tool"]
    assert tool_messages[0].content == "tool result"
    assert isinstance(result.history[-1], AssistantMessage)
    assert result.history[-1].content == "done"


@pytest.mark.asyncio
async def test_run_agent_launches_sub_agent_recursively() -> None:
    launch_call = ToolCall(
        id="launch-1",
        function=FunctionCall(
            name="launch_agent",
            arguments=json.dumps({"task": "Child task", "expert_knowledge": False}),
        ),
    )

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Parent task")],
        model="test-model",
        tools=[],
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[launch_call]),
                AssistantMessage(content="child result"),
                AssistantMessage(content="parent result"),
            ]
        ),
    )

    assert result.status == "awaiting_user"
    tool_messages = [message for message in result.history if message.role == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == "child result"
    assert isinstance(result.history[-1], AssistantMessage)
    assert result.history[-1].content == "parent result"


@pytest.mark.asyncio
async def test_run_agent_applies_tool_policy_before_execution() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments=json.dumps({})),
    )
    tool = MockTool()

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Parent task")],
        model="test-model",
        tools=[tool],
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[external_call]),
                AssistantMessage(content="done"),
            ]
        ),
        tool_policy=DenyPolicy(),
    )

    assert result.status == "awaiting_user"
    assert tool.calls == []
    tool_messages = [message for message in result.history if message.role == "tool"]
    assert tool_messages[0].content == "Tool execution denied."


@pytest.mark.asyncio
async def test_run_agent_compacts_history() -> None:
    compact_call = ToolCall(
        id="compact-1",
        function=FunctionCall(name="compact_conversation", arguments=json.dumps({"summary": "Short summary"})),
    )

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Parent task")],
        model="test-model",
        tools=[],
        completer=ScriptedCompleter(
            [
                AssistantMessage(tool_calls=[compact_call]),
                AssistantMessage(content="done"),
            ]
        ),
    )

    assert result.status == "awaiting_user"
    assert [message.role for message in result.history] == ["system", "user", "tool", "assistant"]
    assert isinstance(result.history[1].content, str)
    assert "Short summary" in result.history[1].content


@pytest.mark.asyncio
async def test_run_agent_returns_failed_status_when_completion_crashes() -> None:
    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Fail")],
        model="test-model",
        tools=[],
        completer=ScriptedCompleter([RuntimeError("boom")]),
    )

    assert result.status == "failed"
    assert result.error == "boom"
