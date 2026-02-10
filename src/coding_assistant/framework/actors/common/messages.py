from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from coding_assistant.framework.actors.common.contracts import MessageSink
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    ProgressCallbacks,
    Tool,
    ToolResult,
    Usage,
)


@dataclass(slots=True)
class LLMCompleteStepRequest:
    request_id: str
    history: list[BaseMessage]
    model: str
    tools: Sequence[Tool]
    progress_callbacks: ProgressCallbacks
    completer: Completer
    reply_to: MessageSink["LLMCompleteStepResponse"]


@dataclass(slots=True)
class LLMCompleteStepResponse:
    request_id: str
    message: AssistantMessage | None
    usage: Usage | None
    error: BaseException | None = None


@dataclass(slots=True)
class HandleToolCallsRequest:
    request_id: str
    message: AssistantMessage
    history: list[BaseMessage]
    task_created_callback: Callable[[str, asyncio.Task[object]], None] | None
    handle_tool_result: Callable[[ToolResult], str] | None
    reply_to: MessageSink["HandleToolCallsResponse"]


@dataclass(slots=True)
class HandleToolCallsResponse:
    request_id: str
    error: BaseException | None = None


@dataclass(slots=True)
class AskRequest:
    request_id: str
    prompt_text: str
    default: str | None
    reply_to: MessageSink["AskResponse"]


@dataclass(slots=True)
class AskResponse:
    request_id: str
    value: str | None
    error: BaseException | None = None


@dataclass(slots=True)
class ConfirmRequest:
    request_id: str
    prompt_text: str
    reply_to: MessageSink["ConfirmResponse"]


@dataclass(slots=True)
class ConfirmResponse:
    request_id: str
    value: bool | None
    error: BaseException | None = None


@dataclass(slots=True)
class PromptRequest:
    request_id: str
    words: list[str] | None
    reply_to: MessageSink["PromptResponse"]


@dataclass(slots=True)
class PromptResponse:
    request_id: str
    value: str | None
    error: BaseException | None = None
