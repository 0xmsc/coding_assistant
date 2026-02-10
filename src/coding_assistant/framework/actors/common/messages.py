from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

from coding_assistant.framework.actors.common.contracts import MessageSink
from coding_assistant.framework.types import AgentContext
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Tool,
    ToolResult,
    Usage,
)


@dataclass(slots=True)
class LLMCompleteStepRequest:
    request_id: str
    history: tuple[BaseMessage, ...]
    model: str
    tools: Sequence[Tool]
    reply_to: MessageSink["LLMCompleteStepResponse"]


@dataclass(slots=True)
class LLMCompleteStepResponse:
    request_id: str
    message: AssistantMessage | None
    usage: Usage | None
    error: BaseException | None = None


@dataclass(slots=True)
class RunAgentRequest:
    request_id: str
    ctx: AgentContext
    tools: Sequence[Tool]
    compact_conversation_at_tokens: int


@dataclass(slots=True)
class RunChatRequest:
    request_id: str
    model: str
    tools: tuple[Tool, ...]
    instructions: str | None
    context_name: str


@dataclass(slots=True)
class RunCompleted:
    request_id: str


@dataclass(slots=True)
class RunFailed:
    request_id: str
    error: BaseException


@dataclass(slots=True)
class HandleToolCallsRequest:
    request_id: str
    message: AssistantMessage
    reply_to: MessageSink["HandleToolCallsResponse"]


@dataclass(slots=True)
class ConfigureToolSetRequest:
    tools: tuple[Tool, ...]


@dataclass(slots=True)
class HandleToolCallsResponse:
    request_id: str
    results: list["ToolCallExecutionResult"]
    cancelled: bool = False
    error: BaseException | None = None


@dataclass(slots=True)
class CancelToolCallsRequest:
    request_id: str


@dataclass(slots=True)
class ToolCallExecutionResult:
    tool_call_id: str
    name: str
    arguments: dict[str, object]
    result: ToolResult
    cancelled: bool = False


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
class UserTextSubmitted:
    text: str


@dataclass(slots=True)
class SessionExitRequested:
    pass


@dataclass(slots=True)
class ClearHistoryRequested:
    pass


@dataclass(slots=True)
class CompactionRequested:
    pass


@dataclass(slots=True)
class ImageAttachRequested:
    source: str | None


@dataclass(slots=True)
class HelpRequested:
    pass


@dataclass(slots=True)
class UserInputFailed:
    error: BaseException


ChatPromptInput: TypeAlias = (
    UserTextSubmitted
    | SessionExitRequested
    | ClearHistoryRequested
    | CompactionRequested
    | ImageAttachRequested
    | HelpRequested
    | UserInputFailed
)


@dataclass(slots=True)
class AgentYieldedToUser:
    request_id: str
    words: list[str] | None
    reply_to: MessageSink[ChatPromptInput]


@dataclass(slots=True)
class PromptRequest:
    request_id: str
    words: list[str] | None
    reply_to: MessageSink["PromptResponse"]
    parse_chat_intent: bool = False


@dataclass(slots=True)
class PromptResponse:
    request_id: str
    value: str | ChatPromptInput | None
    error: BaseException | None = None
