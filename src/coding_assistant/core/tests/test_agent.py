from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from coding_assistant.core.agent import run_agent_event_stream
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.tool_calls import ToolCallExecutionCompleted, stream_tool_call_execution
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompactConversationResult,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    TextToolResult,
    Tool,
    ToolCall,
    ToolMessage,
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
            yield ContentDeltaEvent(content=action.content)

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

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        del parameters
        return TextToolResult(content=self._result)


class ErrorTool(Tool):
    def name(self) -> str:
        return "error_tool"

    def description(self) -> str:
        return "Tool that always fails"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        del parameters
        raise RuntimeError("tool boom")


class CompactingTool(Tool):
    def __init__(self, name: str = "custom_compactor", *, summary: str = "summary text") -> None:
        self._name = name
        self._summary = summary

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Tool that compacts the transcript"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> CompactConversationResult:
        del parameters
        return CompactConversationResult(summary=self._summary)


class NonTextTool(Tool):
    def name(self) -> str:
        return "structured_tool"

    def description(self) -> str:
        return "Tool that returns structured data"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> Any:
        del parameters
        return {"status": "ok"}


def make_system_history() -> list[BaseMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


async def _execute_tool_boundary(
    *,
    boundary: AwaitingToolCalls,
    tools: list[Tool],
) -> list[BaseMessage]:
    completed_history: list[BaseMessage] | None = None
    async for item in stream_tool_call_execution(boundary=boundary, tools=tools):
        if isinstance(item, ToolCallExecutionCompleted):
            completed_history = item.history

    if completed_history is None:
        raise RuntimeError("Tool execution stopped without returning history.")
    return completed_history


@pytest.mark.asyncio
async def test_run_agent_event_stream_yields_existing_boundary_without_pending_model_turn() -> None:
    history = make_system_history()

    events = [
        event
        async for event in run_agent_event_stream(
            history=history,
            model="test-model",
            tools=[],
        )
    ]

    assert events == [AwaitingUser(history=history)]


@pytest.mark.asyncio
async def test_run_agent_event_stream_yields_content_completion_and_boundary() -> None:
    history = [*make_system_history(), UserMessage(content="Hi")]

    events = [
        event
        async for event in run_agent_event_stream(
            history=history,
            model="test-model",
            tools=[],
            streamer=ScriptedStreamer([AssistantMessage(content="Hello from the assistant")]),
        )
    ]

    assert events == [
        ContentDeltaEvent(content="Hello from the assistant"),
        CompletionEvent(
            completion=Completion(
                message=AssistantMessage(content="Hello from the assistant"),
                usage=Usage(tokens=10, cost=0.0),
            ),
        ),
        AwaitingUser(
            history=[
                *make_system_history(),
                UserMessage(content="Hi"),
                AssistantMessage(content="Hello from the assistant"),
            ],
        ),
    ]
    assert history == [*make_system_history(), UserMessage(content="Hi")]


@pytest.mark.asyncio
async def test_run_agent_event_stream_yields_tool_boundary_without_executing() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )

    events = [
        event
        async for event in run_agent_event_stream(
            history=[*make_system_history(), UserMessage(content="Finish the task")],
            model="test-model",
            tools=[MockTool()],
            streamer=ScriptedStreamer([AssistantMessage(tool_calls=[external_call])]),
        )
    ]

    assert events == [
        CompletionEvent(
            completion=Completion(
                message=AssistantMessage(tool_calls=[external_call]),
                usage=Usage(tokens=10, cost=0.0),
            ),
        ),
        AwaitingToolCalls(
            history=[
                *make_system_history(),
                UserMessage(content="Finish the task"),
                AssistantMessage(tool_calls=[external_call]),
            ],
        ),
    ]


@pytest.mark.asyncio
async def test_run_agent_event_stream_recovers_pending_tool_boundary_from_history() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )
    history = [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[external_call]),
    ]

    events = [
        event
        async for event in run_agent_event_stream(
            history=history,
            model="test-model",
            tools=[MockTool()],
        )
    ]

    assert events == [AwaitingToolCalls(history=history)]


@pytest.mark.asyncio
async def test_execute_tool_calls_returns_new_history_without_mutating_boundary_history() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )
    boundary_history = [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[external_call]),
    ]
    boundary = AwaitingToolCalls(
        history=boundary_history,
    )

    result = await _execute_tool_boundary(
        boundary=boundary,
        tools=[MockTool()],
    )

    assert result[-1] == ToolMessage(tool_call_id="call-1", name="mock_tool", content="tool result")
    assert boundary_history == [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[external_call]),
    ]
    assert result is not boundary_history


@pytest.mark.asyncio
async def test_execute_tool_calls_compacts_history_without_orphan_tool_message() -> None:
    compact_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="compact_conversation", arguments='{"summary": "summary text"}'),
    )
    boundary = AwaitingToolCalls(
        history=[
            *make_system_history(),
            UserMessage(content="Finish the task"),
            AssistantMessage(tool_calls=[compact_call]),
        ],
    )

    result = await _execute_tool_boundary(
        boundary=boundary,
        tools=[],
    )

    assert result == [
        *make_system_history(),
        UserMessage(content="A summary of your conversation until now:\n\nsummary text\n\nPlease continue your work."),
    ]


@pytest.mark.asyncio
async def test_execute_tool_calls_stops_after_compaction_result_without_relying_on_tool_name() -> None:
    compact_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="custom_compactor", arguments="{}"),
    )
    later_call = ToolCall(
        id="call-2",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )
    boundary = AwaitingToolCalls(
        history=[
            *make_system_history(),
            UserMessage(content="Finish the task"),
            AssistantMessage(tool_calls=[compact_call, later_call]),
        ],
    )

    result = await _execute_tool_boundary(
        boundary=boundary,
        tools=[CompactingTool(), MockTool()],
    )

    assert result == [
        *make_system_history(),
        UserMessage(content="A summary of your conversation until now:\n\nsummary text\n\nPlease continue your work."),
    ]


@pytest.mark.asyncio
async def test_execute_tool_calls_appends_invalid_tool_result_error() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="structured_tool", arguments="{}"),
    )
    boundary = AwaitingToolCalls(
        history=[
            *make_system_history(),
            UserMessage(content="Finish the task"),
            AssistantMessage(tool_calls=[external_call]),
        ],
    )

    result = await _execute_tool_boundary(
        boundary=boundary,
        tools=[NonTextTool()],
    )

    assert result[-1] == ToolMessage(
        tool_call_id="call-1",
        name="structured_tool",
        content="Error executing tool: Tool 'structured_tool' returned unsupported result type: dict.",
    )


@pytest.mark.asyncio
async def test_execute_tool_calls_appends_tool_execution_error() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="error_tool", arguments="{}"),
    )
    boundary = AwaitingToolCalls(
        history=[
            *make_system_history(),
            UserMessage(content="Finish the task"),
            AssistantMessage(tool_calls=[external_call]),
        ],
    )

    result = await _execute_tool_boundary(
        boundary=boundary,
        tools=[ErrorTool()],
    )

    assert result[-1] == ToolMessage(
        tool_call_id="call-1",
        name="error_tool",
        content="Error executing tool: tool boom",
    )


@pytest.mark.asyncio
async def test_run_agent_event_stream_raises_when_streamer_crashes() -> None:
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in run_agent_event_stream(
            history=[*make_system_history(), UserMessage(content="Fail")],
            model="test-model",
            tools=[],
            streamer=ScriptedStreamer([RuntimeError("boom")]),
        ):
            pass
