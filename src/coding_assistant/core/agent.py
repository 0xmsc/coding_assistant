from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from coding_assistant.core.builtin_tools import CompactConversationTool, RedirectToolCallTool
from coding_assistant.core.history import compact_history
from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompletionEvent,
    LLMEvent,
    Tool,
    ToolCall,
    ToolMessage,
)


@dataclass(frozen=True)
class AwaitingUser:
    """Boundary returned when the caller should provide the next user input."""

    history: list[BaseMessage]


@dataclass(frozen=True)
class AwaitingTools:
    """Boundary returned when the last history entry is an assistant tool-call turn."""

    history: list[BaseMessage]

    @property
    def message(self) -> AssistantMessage:
        """Return the pending assistant message that requested the tool calls."""
        pending_message = _get_pending_tool_message(self.history)
        if pending_message is None:
            raise RuntimeError("AwaitingTools requires the last history entry to be an assistant tool-call message.")
        return pending_message


AgentBoundary = AwaitingUser | AwaitingTools


@dataclass(frozen=True)
class BoundaryEvent:
    """Terminal event emitted when the agent reaches a caller-owned boundary."""

    boundary: AgentBoundary


AgentEvent = LLMEvent | BoundaryEvent


def _parse_tool_call_arguments(tool_call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
    """Decode tool arguments into a JSON object or return a user-visible error."""
    try:
        parsed = json.loads(tool_call.function.arguments)
    except JSONDecodeError as exc:
        return None, f"Error: Tool call arguments `{tool_call.function.arguments}` are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Error: Tool call arguments must decode to a JSON object."
    return parsed, None


def _validate_tools(tools: Sequence[Tool]) -> list[Tool]:
    """Reject duplicate tool names before invoking the model."""
    validated_tools = list(tools)
    names: set[str] = set()
    for tool in validated_tools:
        name = tool.name()
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)

    return validated_tools


def _should_wait_for_user(history: Sequence[BaseMessage]) -> bool:
    """Return true when the transcript already ends in assistant-owned output."""
    return history[-1].role not in {"user", "tool"}


def _get_pending_tool_message(history: Sequence[BaseMessage]) -> AssistantMessage | None:
    """Return the last assistant message when it still has unhandled tool calls."""
    last_message = history[-1]
    if isinstance(last_message, AssistantMessage) and last_message.tool_calls:
        return last_message
    return None


def _build_tools(
    *,
    tools: Sequence[Tool],
) -> list[Tool]:
    """Add built-in tools and wire redirect execution through direct execution."""
    base_tools = [CompactConversationTool(), *tools]
    base_tools_by_name = {tool.name(): tool for tool in base_tools}

    async def execute_redirected_tool(tool_name: str, arguments: dict[str, Any]) -> str:
        target_tool = base_tools_by_name.get(tool_name)
        if target_tool is None:
            raise RuntimeError(f"Tool '{tool_name}' is not available for redirection.")
        return await _execute_resolved_tool(
            tool=target_tool,
            arguments=arguments,
        )

    redirect_tool = RedirectToolCallTool(
        tools=base_tools,
        execute_tool=execute_redirected_tool,
    )
    return _validate_tools([*base_tools, redirect_tool])


async def _execute_resolved_tool(
    *,
    tool: Tool,
    arguments: dict[str, Any],
) -> str:
    """Run one resolved tool and require a text result."""
    result = await tool.execute(arguments)
    if not isinstance(result, str):
        raise TypeError(f"Tool '{tool.name()}' did not return text.")
    return result


async def _execute_tool_call(
    *,
    history: list[BaseMessage],
    tool_call: ToolCall,
    tools_by_name: dict[str, Tool],
) -> tuple[list[BaseMessage], str | None]:
    """Execute one tool call and return any updated history plus result text."""
    tool_name = tool_call.function.name
    if not tool_name:
        raise RuntimeError(f"Tool call {tool_call.id} is missing function name.")

    arguments, parse_error = _parse_tool_call_arguments(tool_call)
    if parse_error is not None:
        return history, parse_error

    assert arguments is not None

    tool = tools_by_name.get(tool_name)
    if tool is None:
        return history, f"Tool '{tool_name}' not found."

    try:
        result = await _execute_resolved_tool(
            tool=tool,
            arguments=arguments,
        )
    except Exception as exc:
        return history, f"Error executing tool: {exc}"

    if tool_name == "compact_conversation":
        compacted_history = compact_history(history, result)
        return compacted_history, None

    return history, result


async def execute_tool_calls(
    *,
    boundary: AwaitingTools,
    tools: Sequence[Tool],
) -> list[BaseMessage]:
    """Execute one tool boundary and append the resulting tool messages."""
    current_history = list(boundary.history)
    all_tools = _build_tools(tools=tools)
    tools_by_name = {tool.name(): tool for tool in all_tools}

    for tool_call in boundary.message.tool_calls:
        current_history, result = await _execute_tool_call(
            history=current_history,
            tool_call=tool_call,
            tools_by_name=tools_by_name,
        )

        if result is None:
            break

        current_history.append(
            ToolMessage(
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
                content=result,
            )
        )

    return current_history


async def run_agent(
    *,
    history: Sequence[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    streamer: Any = openai_stream_completion,
) -> list[BaseMessage]:
    """Advance the transcript until the assistant yields back to the caller."""
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent requires a non-empty history.")

    while True:
        boundary: AgentBoundary | None = None
        async for event in run_agent_event_stream(
            history=current_history,
            model=model,
            tools=tools,
            streamer=streamer,
        ):
            if isinstance(event, BoundaryEvent):
                boundary = event.boundary

        if boundary is None:
            raise RuntimeError("run_agent_event_stream stopped without yielding a boundary.")

        if isinstance(boundary, AwaitingUser):
            return boundary.history

        current_history = await execute_tool_calls(
            boundary=boundary,
            tools=tools,
        )


def _get_boundary(history: list[BaseMessage]) -> AgentBoundary | None:
    """Return the current caller-owned boundary when one already exists."""
    if _get_pending_tool_message(history) is not None:
        return AwaitingTools(history=history)
    if _should_wait_for_user(history):
        return AwaitingUser(history=history)
    return None


async def run_agent_event_stream(
    *,
    history: Sequence[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    streamer: Any = openai_stream_completion,
) -> AsyncIterator[AgentEvent]:
    """Yield streamed LLM events and end with a terminal boundary event."""
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent_event_stream requires a non-empty history.")

    immediate_boundary = _get_boundary(current_history)
    if immediate_boundary is not None:
        yield BoundaryEvent(boundary=immediate_boundary)
        return

    all_tools = _build_tools(tools=tools)
    completion_message: AssistantMessage | None = None
    async for event in streamer(
        current_history,
        model=model,
        tools=all_tools,
    ):
        yield event
        if isinstance(event, CompletionEvent):
            completion_message = event.completion.message

    if completion_message is None:
        raise RuntimeError("Streamer stopped without yielding a completion.")

    current_history.append(completion_message)
    boundary = _get_boundary(current_history)
    if boundary is None:
        raise RuntimeError("run_agent_event_stream did not reach a boundary after the completion.")
    yield BoundaryEvent(boundary=boundary)
