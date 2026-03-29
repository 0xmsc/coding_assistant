from __future__ import annotations

import json
from collections.abc import Sequence
from json import JSONDecodeError
from typing import Any

from coding_assistant.core.boundaries import AwaitingToolCalls
from coding_assistant.core.builtin_tools import CompactConversationTool, RedirectToolCallTool
from coding_assistant.core.history import compact_history
from coding_assistant.llm.types import BaseMessage, Tool, ToolCall, ToolMessage


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


def build_tools(
    *,
    tools: Sequence[Tool],
) -> list[Tool]:
    """Add built-in tools and validate the resulting tool set."""
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


async def execute_tool_calls(
    *,
    boundary: AwaitingToolCalls,
    tools: Sequence[Tool],
) -> list[BaseMessage]:
    """Execute one tool-call boundary and append the resulting tool messages."""
    current_history = list(boundary.history)
    all_tools = build_tools(tools=tools)
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
