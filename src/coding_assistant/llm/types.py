from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Protocol

from dacite import from_dict


class ToolDefinition(Protocol):
    """Interface for tool metadata exposed to the model."""

    def name(self) -> str:
        """Return the stable tool name exposed to the model."""
        ...

    def description(self) -> str:
        """Describe when and why the model should use the tool."""
        ...

    def parameters(self) -> dict[str, Any]:
        """Return the JSON schema for the tool arguments."""
        ...


@dataclass(frozen=True)
class TextToolResult:
    """Normal text output that should be appended as a tool message."""

    content: str


@dataclass(frozen=True)
class CompactConversationResult:
    """A request to compact transcript history around a summary."""

    summary: str


ToolResult = TextToolResult | CompactConversationResult


class Tool(ToolDefinition, ABC):
    """Executable tool implementation."""

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> ToolResult:
        """Execute the tool and return a runtime tool result."""
        ...


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: FunctionCall
    type: str = "function"


@dataclass(frozen=True, kw_only=True)
class BaseMessage:
    role: str
    content: Optional[str | list[dict[str, Any]]] = None
    name: Optional[str] = None


@dataclass(frozen=True, kw_only=True)
class SystemMessage(BaseMessage):
    content: str | list[dict[str, Any]]
    role: Literal["system"] = "system"


@dataclass(frozen=True, kw_only=True)
class UserMessage(BaseMessage):
    content: str | list[dict[str, Any]]
    role: Literal["user"] = "user"


@dataclass(frozen=True, kw_only=True)
class AssistantMessage(BaseMessage):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: Literal["assistant"] = "assistant"
    provider_specific_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ToolMessage(BaseMessage):
    content: str
    tool_call_id: str
    role: Literal["tool"] = "tool"


class StatusLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Usage:
    tokens: int
    cost: float


@dataclass(frozen=True)
class Completion:
    message: AssistantMessage
    usage: Optional[Usage] = None


@dataclass(frozen=True)
class ContentDeltaEvent:
    """One streamed content chunk from the LLM."""

    content: str


@dataclass(frozen=True)
class ReasoningDeltaEvent:
    """One streamed reasoning chunk from the LLM."""

    content: str


@dataclass(frozen=True)
class StatusEvent:
    """One non-content status update from the LLM layer."""

    message: str
    level: StatusLevel = StatusLevel.INFO


@dataclass(frozen=True)
class CompletionEvent:
    """Final completion payload after all chunks have been read."""

    completion: Completion


def message_from_dict(d: dict[str, Any]) -> BaseMessage:
    """Deserialize a stored message dictionary into its typed message class."""
    role = d["role"]

    d = {k: v for k, v in d.items() if v is not None}

    if role == "system":
        return from_dict(data_class=SystemMessage, data=d)
    if role == "user":
        return from_dict(data_class=UserMessage, data=d)
    if role == "assistant":
        return from_dict(data_class=AssistantMessage, data=d)
    if role == "tool":
        return from_dict(data_class=ToolMessage, data=d)
    raise ValueError(f"Unknown role: {role}")


def message_to_dict(msg: BaseMessage) -> dict[str, Any]:
    """Serialize a message while dropping empty optional fields."""

    def factory(data: list[tuple[str, Any]]) -> dict[str, Any]:
        return {k: v for k, v in data if v is not None and v != [] and v != {}}

    return asdict(msg, dict_factory=factory)
