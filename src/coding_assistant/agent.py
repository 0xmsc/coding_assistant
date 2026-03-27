from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from typing import Any

from coding_assistant.history import compact_history
from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompletionEvent,
    ContentDeltaEvent,
    Tool,
    ToolCall,
    ToolMessage,
)
from coding_assistant.tool_policy import DirectToolExecutor, ToolApproved, ToolDenied, ToolExecutor
from coding_assistant.tools.builtin import CompactConversationTool, RedirectToolCallTool


@dataclass(frozen=True)
class AwaitingUser:
    """Boundary returned when the caller should provide the next user input."""

    history: list[BaseMessage]


@dataclass(frozen=True)
class AwaitingTools:
    """Boundary returned when the caller should handle assistant tool calls."""

    history: list[BaseMessage]
    message: AssistantMessage


AgentBoundary = AwaitingUser | AwaitingTools


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


def _build_tools(
    *,
    tools: Sequence[Tool],
    tool_executor: ToolExecutor,
) -> list[Tool]:
    """Add built-in tools and wire redirect execution through the executor."""
    base_tools = [CompactConversationTool(), *tools]
    base_tools_by_name = {tool.name(): tool for tool in base_tools}

    async def execute_redirected_tool(tool_name: str, arguments: dict[str, Any]) -> ToolApproved | ToolDenied:
        target_tool = base_tools_by_name.get(tool_name)
        if target_tool is None:
            raise RuntimeError(f"Tool '{tool_name}' is not available for redirection.")
        return await tool_executor.execute(
            tool_call_id="",
            tool=target_tool,
            arguments=arguments,
        )

    redirect_tool = RedirectToolCallTool(
        tools=base_tools,
        execute_tool=execute_redirected_tool,
    )
    return _validate_tools([*base_tools, redirect_tool])


async def _consume_completion(
    *,
    history: list[BaseMessage],
    tools: list[Tool],
    model: str,
    streamer: Any,
    on_content: Callable[[str], None] | None,
) -> AssistantMessage:
    """Read one streamed completion and return the final assistant message."""
    completion_message: AssistantMessage | None = None
    async for event in streamer(
        history,
        model=model,
        tools=tools,
    ):
        if isinstance(event, ContentDeltaEvent) and on_content is not None:
            on_content(event.content)
        elif isinstance(event, CompletionEvent):
            completion_message = event.completion.message

    if completion_message is None:
        raise RuntimeError("Streamer stopped without yielding a completion.")
    return completion_message


async def _execute_tool_call(
    *,
    history: list[BaseMessage],
    tool_call: ToolCall,
    tools_by_name: dict[str, Tool],
    tool_executor: ToolExecutor,
) -> tuple[list[BaseMessage], str]:
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
        execution_result = await tool_executor.execute(
            tool_call_id=tool_call.id,
            tool=tool,
            arguments=arguments,
        )
    except Exception as exc:
        return history, f"Error executing tool: {exc}"

    if isinstance(execution_result, ToolDenied):
        return history, execution_result.content

    assert isinstance(execution_result, ToolApproved)
    if tool_name == "compact_conversation":
        compacted_history = compact_history(history, execution_result.content)
        return compacted_history, "Conversation compacted and history reset."

    return history, execution_result.content


async def run_agent_until_boundary(
    *,
    history: Sequence[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    streamer: Any = openai_stream_completion,
    on_content: Callable[[str], None] | None = None,
) -> AgentBoundary:
    """Advance the transcript until the next explicit caller-owned boundary."""
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent_until_boundary requires a non-empty history.")

    if _should_wait_for_user(current_history):
        return AwaitingUser(history=current_history)

    all_tools = _build_tools(tools=tools, tool_executor=DirectToolExecutor())
    message = await _consume_completion(
        history=current_history,
        tools=all_tools,
        model=model,
        streamer=streamer,
        on_content=on_content,
    )
    current_history.append(message)
    if message.tool_calls:
        return AwaitingTools(history=current_history, message=message)
    return AwaitingUser(history=current_history)


async def execute_tool_calls(
    *,
    boundary: AwaitingTools,
    tools: Sequence[Tool],
    tool_executor: ToolExecutor | None = None,
) -> list[BaseMessage]:
    """Execute one tool boundary and append the resulting tool messages."""
    current_history = list(boundary.history)
    resolved_tool_executor = tool_executor or DirectToolExecutor()
    all_tools = _build_tools(tools=tools, tool_executor=resolved_tool_executor)
    tools_by_name = {tool.name(): tool for tool in all_tools}

    for tool_call in boundary.message.tool_calls:
        current_history, result = await _execute_tool_call(
            history=current_history,
            tool_call=tool_call,
            tools_by_name=tools_by_name,
            tool_executor=resolved_tool_executor,
        )
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
    tool_executor: ToolExecutor | None = None,
    streamer: Any = openai_stream_completion,
    on_content: Callable[[str], None] | None = None,
) -> list[BaseMessage]:
    """Advance the transcript until the assistant yields back to the caller."""
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent requires a non-empty history.")

    while True:
        boundary = await run_agent_until_boundary(
            history=current_history,
            model=model,
            tools=tools,
            streamer=streamer,
            on_content=on_content,
        )
        if isinstance(boundary, AwaitingUser):
            return boundary.history

        current_history = await execute_tool_calls(
            boundary=boundary,
            tools=tools,
            tool_executor=tool_executor,
        )
