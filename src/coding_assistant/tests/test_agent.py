from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from coding_assistant import execute_tool_calls, run_agent, run_agent_until_boundary
from coding_assistant.agent import AwaitingTools, AwaitingUser
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent as LLMContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from coding_assistant.tool_policy import ToolApproved, ToolDenied


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
    def __init__(self, name: str = "mock_tool", *, result: str = "tool result") -> None:
        self._name = name
        self._result = result

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Mock external tool"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> str:
        return self._result


class ErrorTool(Tool):
    def name(self) -> str:
        return "error_tool"

    def description(self) -> str:
        return "Tool that always fails"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> str:
        del parameters
        raise RuntimeError("tool boom")


class DenyingToolExecutor:
    async def execute(
        self,
        *,
        tool_call_id: str,
        tool: Tool,
        arguments: dict[str, Any],
    ) -> ToolApproved | ToolDenied:
        del tool_call_id, tool, arguments
        return ToolDenied(content="Tool execution denied.")


def make_system_history() -> list[BaseMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


@pytest.mark.asyncio
async def test_run_agent_returns_existing_history_without_pending_model_turn() -> None:
    history = make_system_history()

    result = await run_agent(history=history, model="test-model", tools=[])

    assert result == history


@pytest.mark.asyncio
async def test_run_agent_until_boundary_returns_existing_history_without_pending_model_turn() -> None:
    history = make_system_history()

    result = await run_agent_until_boundary(history=history, model="test-model", tools=[])

    assert result == AwaitingUser(history=history)


@pytest.mark.asyncio
async def test_run_agent_until_boundary_streams_content_and_returns_awaiting_user() -> None:
    chunks: list[str] = []

    result = await run_agent_until_boundary(
        history=[*make_system_history(), UserMessage(content="Hi")],
        model="test-model",
        tools=[],
        streamer=ScriptedStreamer([AssistantMessage(content="Hello from the assistant")]),
        on_content=chunks.append,
    )

    assert chunks == ["Hello from the assistant"]
    assert result == AwaitingUser(
        history=[
            *make_system_history(),
            UserMessage(content="Hi"),
            AssistantMessage(content="Hello from the assistant"),
        ]
    )


@pytest.mark.asyncio
async def test_run_agent_until_boundary_returns_tool_boundary_without_executing() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )

    result = await run_agent_until_boundary(
        history=[*make_system_history(), UserMessage(content="Finish the task")],
        model="test-model",
        tools=[MockTool()],
        streamer=ScriptedStreamer([AssistantMessage(tool_calls=[external_call])]),
    )

    assert result == AwaitingTools(
        history=[
            *make_system_history(),
            UserMessage(content="Finish the task"),
            AssistantMessage(tool_calls=[external_call]),
        ],
        message=AssistantMessage(tool_calls=[external_call]),
    )


@pytest.mark.asyncio
async def test_execute_tool_calls_appends_tool_messages_for_a_boundary() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )
    boundary = AwaitingTools(
        history=[
            *make_system_history(),
            UserMessage(content="Finish the task"),
            AssistantMessage(tool_calls=[external_call]),
        ],
        message=AssistantMessage(tool_calls=[external_call]),
    )

    result = await execute_tool_calls(
        boundary=boundary,
        tools=[MockTool()],
    )

    assert result[-1] == ToolMessage(tool_call_id="call-1", name="mock_tool", content="tool result")


@pytest.mark.asyncio
async def test_run_agent_streams_content_and_appends_final_message() -> None:
    chunks: list[str] = []

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Hi")],
        model="test-model",
        tools=[],
        streamer=ScriptedStreamer([AssistantMessage(content="Hello from the assistant")]),
        on_content=chunks.append,
    )

    assert chunks == ["Hello from the assistant"]
    assert result[-1] == AssistantMessage(content="Hello from the assistant")


@pytest.mark.asyncio
async def test_run_agent_executes_tool_calls_and_continues() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Finish the task")],
        model="test-model",
        tools=[MockTool()],
        streamer=ScriptedStreamer(
            [
                AssistantMessage(tool_calls=[external_call]),
                AssistantMessage(content="Finished after the tool call"),
            ]
        ),
    )

    assert result[-3] == AssistantMessage(tool_calls=[external_call])
    assert result[-2] == ToolMessage(tool_call_id="call-1", name="mock_tool", content="tool result")
    assert result[-1] == AssistantMessage(content="Finished after the tool call")


@pytest.mark.asyncio
async def test_run_agent_appends_denied_tool_result_and_continues() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Finish the task")],
        model="test-model",
        tools=[MockTool()],
        tool_executor=DenyingToolExecutor(),
        streamer=ScriptedStreamer(
            [
                AssistantMessage(tool_calls=[external_call]),
                AssistantMessage(content="I can continue without that tool."),
            ]
        ),
    )

    assert result[-2] == ToolMessage(tool_call_id="call-1", name="mock_tool", content="Tool execution denied.")
    assert result[-1] == AssistantMessage(content="I can continue without that tool.")


@pytest.mark.asyncio
async def test_run_agent_appends_tool_execution_errors_and_continues() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="error_tool", arguments="{}"),
    )

    result = await run_agent(
        history=[*make_system_history(), UserMessage(content="Finish the task")],
        model="test-model",
        tools=[ErrorTool()],
        streamer=ScriptedStreamer(
            [
                AssistantMessage(tool_calls=[external_call]),
                AssistantMessage(content="I recovered from the tool error."),
            ]
        ),
    )

    assert result[-2] == ToolMessage(
        tool_call_id="call-1", name="error_tool", content="Error executing tool: tool boom"
    )
    assert result[-1] == AssistantMessage(content="I recovered from the tool error.")


@pytest.mark.asyncio
async def test_run_agent_raises_when_streamer_crashes() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        await run_agent(
            history=[*make_system_history(), UserMessage(content="Fail")],
            model="test-model",
            tools=[],
            streamer=ScriptedStreamer([RuntimeError("boom")]),
        )
