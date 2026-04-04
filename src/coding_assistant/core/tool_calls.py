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
    TextToolResult,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
)

ToolCallKind = Literal["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "other"]


@dataclass(frozen=True)
class ToolCallLifecycleEvent:
    tool_call_id: str
    tool_name: str
    title: str
    kind: ToolCallKind
    status: Literal["in_progress", "completed", "failed"]
    raw_input: dict[str, Any] | None = None
    raw_output: Any | None = None
    content: str | None = None


@dataclass(frozen=True)
class ToolCallExecutionCompleted:
    history: list[BaseMessage]


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


def _tool_call_kind(tool_name: str) -> ToolCallKind:
    normalized = tool_name.lower()
    if any(part in normalized for part in ("read", "cat", "get_output", "skills_read")):
        return "read"
    if any(part in normalized for part in ("write", "edit", "todo_add", "todo_complete")):
        return "edit"
    if any(part in normalized for part in ("delete", "remove")):
        return "delete"
    if any(part in normalized for part in ("move", "rename")):
        return "move"
    if any(part in normalized for part in ("search", "list")):
        return "search"
    if any(part in normalized for part in ("shell", "python", "task", "execute", "kill")):
        return "execute"
    if "fetch" in normalized:
        return "fetch"
    if "think" in normalized or "compact" in normalized:
        return "think"
    return "other"


def _tool_call_title(tool_name: str) -> str:
    return tool_name or "tool_call"


def _trace_tool_call_event(event: ToolCallLifecycleEvent) -> None:
    """Write a tool call lifecycle event to the trace log when tracing is enabled."""
    if not trace_enabled():
        return
    trace_json(
        f"tool_{event.tool_call_id}_{event.status}.json5",
        {
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "kind": event.kind,
            "status": event.status,
            "raw_input": event.raw_input,
            "raw_output": event.raw_output,
        },
    )


def _record_tool_call_event(
    *,
    tool_call: ToolCall,
    status: Literal["in_progress", "completed", "failed"],
    raw_input: dict[str, Any] | None = None,
    raw_output: Any | None = None,
    content: str | None = None,
) -> ToolCallLifecycleEvent:
    event = ToolCallLifecycleEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.function.name,
        title=_tool_call_title(tool_call.function.name),
        kind=_tool_call_kind(tool_call.function.name),
        status=status,
        raw_input=raw_input,
        raw_output=raw_output,
        content=content,
    )
    _trace_tool_call_event(event)
    return event


def _append_tool_message(
    history: list[BaseMessage],
    *,
    tool_call: ToolCall,
    content: str,
) -> None:
    history.append(
        ToolMessage(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            content=content,
        ),
    )


def _append_failed_tool_call(
    history: list[BaseMessage],
    *,
    tool_call: ToolCall,
    content: str,
    raw_input: dict[str, Any] | None = None,
) -> ToolCallLifecycleEvent:
    event = _record_tool_call_event(
        tool_call=tool_call,
        status="failed",
        raw_input=raw_input,
        content=content,
    )
    _append_tool_message(history, tool_call=tool_call, content=content)
    return event


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
                raw_input=arguments,
            ),
        )
    return events


def _tool_result_content(result: ToolResult) -> str:
    if isinstance(result, TextToolResult):
        return result.content
    return result.summary


def _tool_result_raw_output(result: ToolResult) -> Any:
    if isinstance(result, TextToolResult):
        return result.content
    return {"summary": result.summary}


def _apply_tool_result(
    history: list[BaseMessage],
    *,
    tool_call: ToolCall,
    result: ToolResult,
) -> tuple[list[BaseMessage], bool]:
    if isinstance(result, CompactConversationResult):
        return compact_history(history, result.summary), True
    _append_tool_message(history, tool_call=tool_call, content=result.content)
    return history, False


async def stream_tool_call_execution(
    *,
    boundary: AwaitingToolCalls,
    tools: Sequence[Tool],
) -> AsyncIterator[ToolCallLifecycleEvent | ToolCallExecutionCompleted]:
    """Yield lifecycle updates while executing one tool-call boundary."""
    current_history = list(boundary.history)
    all_tools = build_tools(tools=tools)
    tools_by_name = {tool.name(): tool for tool in all_tools}

    for index, tool_call in enumerate(boundary.message.tool_calls):
        tool_name = tool_call.function.name
        arguments, parse_error = _parse_tool_call_arguments(tool_call)

        if parse_error is not None:
            yield _append_failed_tool_call(
                current_history,
                tool_call=tool_call,
                content=parse_error,
            )
            continue

        assert arguments is not None
        tool = tools_by_name.get(tool_name)
        if tool is None:
            error = f"Tool '{tool_name}' not found."
            yield _append_failed_tool_call(
                current_history,
                tool_call=tool_call,
                content=error,
                raw_input=arguments,
            )
            continue

        yield _record_tool_call_event(
            tool_call=tool_call,
            status="in_progress",
            raw_input=arguments,
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
            yield _append_failed_tool_call(
                current_history,
                tool_call=tool_call,
                content=error,
                raw_input=arguments,
            )
            continue

        yield _record_tool_call_event(
            tool_call=tool_call,
            status="completed",
            raw_input=arguments,
            raw_output=_tool_result_raw_output(result),
            content=_tool_result_content(result),
        )

        current_history, should_stop = _apply_tool_result(
            current_history,
            tool_call=tool_call,
            result=result,
        )
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
    if isinstance(result, (TextToolResult, CompactConversationResult)):
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
