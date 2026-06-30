from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any, Literal

from coding_assistant.core.boundaries import AwaitingToolCalls
from coding_assistant.core.builtin_tools import CompactConversationTool, RedirectToolCallTool
from coding_assistant.core.history import compact_history
from coding_assistant.infra.trace import trace_enabled, trace_json
from coding_assistant.llm.types import (
    BaseMessage,
    CompactConversationResult,
    MessageContent,
    TextToolResult,
    Tool,
    ToolCall,
    ToolMessageResult,
    ToolMessage,
    ToolResult,
)


@dataclass(frozen=True)
class ToolCallLifecycleEvent:
    tool_call_id: str
    tool_name: str
    status: Literal["in_progress", "completed", "failed"]
    arguments: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class ToolCallExecutionCompleted:
    history: list[BaseMessage]


@dataclass(frozen=True)
class ToolMessageProduced:
    message: ToolMessage


TOOL_CANCELLED_MESSAGE = "Tool execution cancelled by user before completion."
TOOL_NOT_STARTED_MESSAGE = "Tool execution was not started because the run was cancelled by user."


@dataclass(frozen=True)
class ToolExecutionCancelled(Exception):
    history: list[BaseMessage]

    def __str__(self) -> str:
        return "Tool execution cancelled."


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


def _trace_tool_call_event(event: ToolCallLifecycleEvent) -> None:
    """Write a tool call lifecycle event to the trace log when tracing is enabled."""
    if not trace_enabled():
        return
    trace_json(
        f"tool_{event.tool_call_id}_{event.status}.json5",
        {
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "status": event.status,
            "arguments": event.arguments,
            "error": event.error,
        },
    )


def _record_tool_call_event(
    *,
    tool_call: ToolCall,
    status: Literal["in_progress", "completed", "failed"],
    arguments: dict[str, Any] | None = None,
    error: str | None = None,
) -> ToolCallLifecycleEvent:
    event = ToolCallLifecycleEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.function.name,
        status=status,
        arguments=arguments,
        error=error,
    )
    _trace_tool_call_event(event)
    return event


def _append_tool_message(
    history: list[BaseMessage],
    *,
    tool_call: ToolCall,
    content: MessageContent,
) -> ToolMessage:
    message = ToolMessage(
        tool_call_id=tool_call.id,
        name=tool_call.function.name,
        content=content,
    )
    history.append(message)
    return message


def _append_failed_tool_call(
    history: list[BaseMessage],
    *,
    tool_call: ToolCall,
    content: str,
    arguments: dict[str, Any] | None = None,
) -> tuple[ToolCallLifecycleEvent, ToolMessage]:
    event = _record_tool_call_event(
        tool_call=tool_call,
        status="failed",
        arguments=arguments,
        error=content,
    )
    message = _append_tool_message(history, tool_call=tool_call, content=content)
    return event, message


def _append_cancelled_tool_messages(
    history: list[BaseMessage],
    *,
    tool_calls: Sequence[ToolCall],
    cancelled_at_index: int,
) -> list[ToolCallLifecycleEvent]:
    events: list[ToolCallLifecycleEvent] = []
    for index, tool_call in enumerate(tool_calls[cancelled_at_index:], start=cancelled_at_index):
        arguments, _ = _parse_tool_call_arguments(tool_call)
        content = TOOL_CANCELLED_MESSAGE if index == cancelled_at_index else TOOL_NOT_STARTED_MESSAGE
        events.append(
            _append_failed_tool_call(
                history,
                tool_call=tool_call,
                content=content,
                arguments=arguments,
            )[0],
        )
    return events


def _apply_tool_result(
    history: list[BaseMessage],
    *,
    tool_call: ToolCall,
    result: ToolResult,
) -> tuple[list[BaseMessage], ToolMessage | None, bool]:
    if isinstance(result, CompactConversationResult):
        return compact_history(history, result.summary), None, True
    message = _append_tool_message(history, tool_call=tool_call, content=result.content)
    return history, message, False


async def stream_tool_call_execution(
    *,
    boundary: AwaitingToolCalls,
    tools: Sequence[Tool],
) -> AsyncIterator[ToolCallLifecycleEvent | ToolMessageProduced | ToolCallExecutionCompleted]:
    """Yield lifecycle updates while executing one tool-call boundary."""
    current_history = list(boundary.history)
    all_tools = build_tools(tools=tools)
    tools_by_name = {tool.name(): tool for tool in all_tools}

    for index, tool_call in enumerate(boundary.message.tool_calls):
        tool_name = tool_call.function.name
        arguments, parse_error = _parse_tool_call_arguments(tool_call)

        if parse_error is not None:
            event, failed_message = _append_failed_tool_call(
                current_history,
                tool_call=tool_call,
                content=parse_error,
            )
            yield event
            yield ToolMessageProduced(message=failed_message)
            continue

        assert arguments is not None
        tool = tools_by_name.get(tool_name)
        if tool is None:
            error = f"Tool '{tool_name}' not found."
            event, failed_message = _append_failed_tool_call(
                current_history,
                tool_call=tool_call,
                content=error,
                arguments=arguments,
            )
            yield event
            yield ToolMessageProduced(message=failed_message)
            continue

        yield _record_tool_call_event(
            tool_call=tool_call,
            status="in_progress",
            arguments=arguments,
        )

        try:
            result = await _execute_resolved_tool(
                tool=tool,
                arguments=arguments,
            )
        except asyncio.CancelledError as exc:
            for event in _append_cancelled_tool_messages(
                current_history,
                tool_calls=boundary.message.tool_calls,
                cancelled_at_index=index,
            ):
                yield event
            raise ToolExecutionCancelled(history=current_history) from exc
        except Exception as exc:
            error = f"Error executing tool: {exc}"
            event, failed_message = _append_failed_tool_call(
                current_history,
                tool_call=tool_call,
                content=error,
                arguments=arguments,
            )
            yield event
            yield ToolMessageProduced(message=failed_message)
            continue

        yield _record_tool_call_event(
            tool_call=tool_call,
            status="completed",
            arguments=arguments,
        )

        current_history, tool_message, should_stop = _apply_tool_result(
            current_history,
            tool_call=tool_call,
            result=result,
        )
        if tool_message is not None:
            yield ToolMessageProduced(message=tool_message)
        if should_stop:
            break

    yield ToolCallExecutionCompleted(history=current_history)


async def _execute_resolved_tool(
    *,
    tool: Tool,
    arguments: dict[str, Any],
) -> ToolResult:
    """Run one resolved tool and require a typed tool result."""
    result = await tool.execute(arguments)
    if isinstance(result, (TextToolResult, ToolMessageResult, CompactConversationResult)):
        return result
    raise TypeError(f"Tool '{tool.name()}' returned unsupported result type: {type(result).__name__}.")


def build_tools(
    *,
    tools: Sequence[Tool],
) -> list[Tool]:
    """Add built-in tools and validate the resulting tool set."""
    base_tools = [CompactConversationTool(), *tools]
    base_tools_by_name = {tool.name(): tool for tool in base_tools}

    async def execute_redirected_tool(tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        target_tool = base_tools_by_name.get(tool_name)
        if target_tool is None:
            raise RuntimeError(f"Tool '{tool_name}' is not available for redirection.")
        return await target_tool.execute(arguments)

    redirect_tool = RedirectToolCallTool(
        tools=base_tools,
        execute_tool=execute_redirected_tool,
    )
    return _validate_tools([*base_tools, redirect_tool])
