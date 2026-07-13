from __future__ import annotations

from typing import Any

import pytest

from coding_assistant.core.tool_calls import (
    ToolCallExecutionCompleted,
    ToolCallLifecycleEvent,
    ToolMessageProduced,
    build_tools,
    stream_tool_call_execution,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompactConversationResult,
    FunctionCall,
    SystemMessage,
    TextToolResult,
    Tool,
    ToolCall,
    ToolMessageResult,
    ToolMessage,
    UserMessage,
)


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


class MultimodalTool(Tool):
    def __init__(self, name: str = "multimodal_tool") -> None:
        self._name = name

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Tool that returns multimodal content"

    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, parameters: dict[str, Any]) -> ToolMessageResult:
        del parameters
        return ToolMessageResult(
            content=[
                {"type": "text", "text": "loaded image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        )


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


async def _execute_tool_calls(
    *,
    history: list[BaseMessage],
    tools: list[Tool],
) -> list[BaseMessage]:
    message = history[-1]
    assert isinstance(message, AssistantMessage)
    completed_history: list[BaseMessage] | None = None
    async for item in stream_tool_call_execution(
        history=history,
        message=message,
        tools=build_tools(tools=tools),
    ):
        if isinstance(item, ToolCallExecutionCompleted):
            completed_history = item.history

    if completed_history is None:
        raise RuntimeError("Tool execution stopped without returning history.")
    return completed_history


@pytest.mark.asyncio
async def test_execute_tool_calls_returns_new_history_without_mutating_input() -> None:
    external_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )
    history = [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[external_call]),
    ]
    result = await _execute_tool_calls(
        history=history,
        tools=[MockTool()],
    )

    assert result[-1] == ToolMessage(tool_call_id="call-1", name="mock_tool", content="tool result")
    assert history == [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[external_call]),
    ]
    assert result is not history


@pytest.mark.asyncio
async def test_execute_tool_calls_compacts_history_without_orphan_tool_message() -> None:
    compact_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="compact_conversation", arguments='{"summary": "summary text"}'),
    )
    history = [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[compact_call]),
    ]

    result = await _execute_tool_calls(
        history=history,
        tools=[],
    )

    assert result == [
        *make_system_history(),
        UserMessage(content="A summary of your conversation until now:\n\nsummary text\n\nPlease continue your work."),
    ]


@pytest.mark.asyncio
async def test_execute_tool_calls_appends_multimodal_tool_result_as_tool_message() -> None:
    multimodal_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="multimodal_tool", arguments="{}"),
    )
    later_call = ToolCall(
        id="call-2",
        function=FunctionCall(name="mock_tool", arguments="{}"),
    )
    history = [
        *make_system_history(),
        UserMessage(content="Load it"),
        AssistantMessage(tool_calls=[multimodal_call, later_call]),
    ]

    result = await _execute_tool_calls(
        history=history,
        tools=[MultimodalTool(), MockTool()],
    )

    assert result[-2:] == [
        ToolMessage(
            tool_call_id="call-1",
            name="multimodal_tool",
            content=[
                {"type": "text", "text": "loaded image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        ),
        ToolMessage(tool_call_id="call-2", name="mock_tool", content="tool result"),
    ]


@pytest.mark.asyncio
async def test_execute_tool_calls_streams_multimodal_tool_message() -> None:
    multimodal_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="multimodal_tool", arguments="{}"),
    )
    message = AssistantMessage(tool_calls=[multimodal_call])
    history: list[BaseMessage] = [*make_system_history(), UserMessage(content="Load it"), message]

    events = [
        event
        async for event in stream_tool_call_execution(
            history=history,
            message=message,
            tools=build_tools(tools=[MultimodalTool()]),
        )
    ]

    completed = events[1]
    produced = events[2]
    assert isinstance(completed, ToolCallLifecycleEvent)
    assert completed.status == "completed"
    assert isinstance(produced, ToolMessageProduced)
    assert produced.message.content == [
        {"type": "text", "text": "loaded image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
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
    history = [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[compact_call, later_call]),
    ]

    result = await _execute_tool_calls(
        history=history,
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
    history = [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[external_call]),
    ]

    result = await _execute_tool_calls(
        history=history,
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
    history = [
        *make_system_history(),
        UserMessage(content="Finish the task"),
        AssistantMessage(tool_calls=[external_call]),
    ]

    result = await _execute_tool_calls(
        history=history,
        tools=[ErrorTool()],
    )

    assert result[-1] == ToolMessage(
        tool_call_id="call-1",
        name="error_tool",
        content="Error executing tool: tool boom",
    )
