from __future__ import annotations

import json
from json import JSONDecodeError
from collections.abc import Sequence
from typing import Any

from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    ToolDefinition,
    ToolCall,
    ToolResult,
)
from coding_assistant.runtime.events import AssistantDeltaEvent


class RuntimeProgressCallbacks(NullProgressCallbacks):
    def __init__(self, emit: Any) -> None:
        self._emit = emit

    def on_content_chunk(self, chunk: str) -> None:
        self._emit(AssistantDeltaEvent(delta=chunk))


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
    tools: Sequence[ToolDefinition],
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
