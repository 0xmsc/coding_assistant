from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from coding_assistant.framework.types import AgentContext, AgentOutput
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    ToolResult,
    Usage,
)


@dataclass(slots=True)
class ToolCapability:
    name: str
    uri: str
    description: str
    parameters: dict[str, object]


@dataclass(slots=True)
class LLMCompleteStepRequest:
    request_id: str
    history: tuple[BaseMessage, ...]
    model: str
    tool_capabilities: tuple[ToolCapability, ...]
    reply_to_uri: str


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
    tool_capabilities: tuple[ToolCapability, ...]
    compact_conversation_at_tokens: int
    tool_call_actor_uri: str
    reply_to_uri: str | None = None


@dataclass(slots=True)
class RunChatRequest:
    request_id: str
    history: list[BaseMessage]
    model: str
    tool_capabilities: tuple[ToolCapability, ...]
    instructions: str | None
    context_name: str
    user_actor_uri: str
    tool_call_actor_uri: str
    reply_to_uri: str | None = None


@dataclass(slots=True)
class RunCompleted:
    request_id: str
    history: tuple[BaseMessage, ...] | None = None
    agent_output: AgentOutput | None = None


@dataclass(slots=True)
class RunFailed:
    request_id: str
    error: BaseException


@dataclass(slots=True)
class HandleToolCallsRequest:
    request_id: str
    message: AssistantMessage
    reply_to_uri: str
    tool_capabilities: tuple[ToolCapability, ...] | None = None


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
    reply_to_uri: str


@dataclass(slots=True)
class AskResponse:
    request_id: str
    value: str | None
    error: BaseException | None = None


@dataclass(slots=True)
class ConfirmRequest:
    request_id: str
    prompt_text: str
    reply_to_uri: str


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
    reply_to_uri: str


@dataclass(slots=True)
class PromptRequest:
    request_id: str
    words: list[str] | None
    reply_to_uri: str
    parse_chat_intent: bool = False


@dataclass(slots=True)
class PromptResponse:
    request_id: str
    value: str | ChatPromptInput | None
    error: BaseException | None = None
