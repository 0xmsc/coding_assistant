from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any

from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
    ToolCall,
    ToolResult,
)
from coding_assistant.runtime.builtin_tools import CompactConversationTool, FinishTaskTool
from coding_assistant.runtime.events import AssistantDeltaEvent


CHAT_START_MESSAGE_TEMPLATE = """
## General

- You are an agent.
- You are in chat mode.
  - Use tools only when they materially advance the work.
  - When you want the user to reply, write a normal assistant message without tool calls.

{instructions_section}
""".strip()


AGENT_START_MESSAGE_TEMPLATE = """
## General

- You are an agent.
- You are working in agent mode.
  - Use tools when they materially advance the work.
  - When you are done, call the `finish_task` tool.
  - When you need more information, write a normal assistant message without tool calls.

## Task

{task}

{instructions_section}
""".strip()


def create_chat_start_message(instructions: str) -> str:
    return CHAT_START_MESSAGE_TEMPLATE.format(instructions_section=_render_instructions_section(instructions))


def create_agent_start_message(*, task: str, instructions: str) -> str:
    return AGENT_START_MESSAGE_TEMPLATE.format(
        task=task,
        instructions_section=_render_instructions_section(instructions),
    )


def _render_instructions_section(instructions: str) -> str:
    cleaned = instructions.strip()
    if not cleaned:
        return ""
    return f"## Instructions\n\n{cleaned}"


def ensure_builtin_tools(*, tools: list[Tool], include_finish_tool: bool) -> list[Tool]:
    result = list(tools)
    if include_finish_tool and not any(tool.name() == "finish_task" for tool in result):
        result.append(FinishTaskTool())
    if not any(tool.name() == "compact_conversation" for tool in result):
        result.append(CompactConversationTool())
    return result


class RuntimeProgressCallbacks(NullProgressCallbacks):
    def __init__(self, emit: Any) -> None:
        self._emit = emit

    def on_content_chunk(self, chunk: str) -> None:
        self._emit(AssistantDeltaEvent(delta=chunk))


async def execute_tool_call(*, function_name: str, function_args: dict[str, Any], tools: list[Tool]) -> ToolResult:
    for tool in tools:
        if tool.name() == function_name:
            return await tool.execute(function_args)
    raise ValueError(f"Tool {function_name} not found in session tools.")


def parse_tool_call_arguments(tool_call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
    try:
        parsed = json.loads(tool_call.function.arguments)
    except JSONDecodeError as exc:
        return None, f"Error: Tool call arguments `{tool_call.function.arguments}` are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Error: Tool call arguments must decode to a JSON object."
    return parsed, None


def normalize_tool_result(result: ToolResult) -> str:
    if hasattr(result, "content"):
        content = getattr(result, "content")
        if isinstance(content, str):
            return content
    return f"Tool produced result of type {type(result).__name__}"


async def complete_single_step(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    progress_callbacks: ProgressCallbacks,
    completer: Any,
) -> tuple[AssistantMessage, Any]:
    if not history:
        raise RuntimeError("History is required in order to run a step.")

    completion = await completer(
        history,
        model=model,
        tools=tools,
        callbacks=progress_callbacks,
    )
    return completion.message, completion.usage
