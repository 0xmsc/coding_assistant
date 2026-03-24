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


class Tool(ToolDefinition, ABC):
    """Executable tool implementation."""

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> str:
        """Execute the tool and return a text result for the transcript."""
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


class ProgressCallbacks(ABC):
    """Abstract interface for agent callbacks."""

    @abstractmethod
    def on_status_message(self, message: str, level: StatusLevel = StatusLevel.INFO) -> None:
        """Handle status messages from the system."""
        pass

    @abstractmethod
    def on_user_message(self, context_name: str, message: UserMessage, *, force: bool = False) -> None:
        """Handle messages with role: user."""
        pass

    @abstractmethod
    def on_assistant_message(self, context_name: str, message: AssistantMessage, *, force: bool = False) -> None:
        """Handle messages with role: assistant."""
        pass

    @abstractmethod
    def on_tool_start(self, context_name: str, tool_call: ToolCall, arguments: dict[str, Any]) -> None:
        """Handle tool start events."""
        pass

    @abstractmethod
    def on_tool_message(
        self, context_name: str, message: ToolMessage, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Handle messages with role: tool."""
        pass

    @abstractmethod
    def on_content_chunk(self, chunk: str) -> None:
        """Handle LLM content chunks."""
        pass

    @abstractmethod
    def on_reasoning_chunk(self, chunk: str) -> None:
        """Handle LLM reasoning chunks."""
        pass

    @abstractmethod
    def on_chunks_end(self) -> None:
        """Handle end of LLM chunks."""
        pass


class NullProgressCallbacks(ProgressCallbacks):
    """Null object implementation that does nothing."""

    def on_status_message(self, message: str, level: StatusLevel = StatusLevel.INFO) -> None:
        """Ignore status updates."""
        pass

    def on_user_message(self, context_name: str, message: UserMessage, *, force: bool = False) -> None:
        """Ignore user-message notifications."""
        pass

    def on_assistant_message(self, context_name: str, message: AssistantMessage, *, force: bool = False) -> None:
        """Ignore assistant-message notifications."""
        pass

    def on_tool_start(self, context_name: str, tool_call: ToolCall, arguments: dict[str, Any]) -> None:
        """Ignore tool-start notifications."""
        pass

    def on_tool_message(
        self, context_name: str, message: ToolMessage, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """Ignore tool result notifications."""
        pass

    def on_content_chunk(self, chunk: str) -> None:
        """Ignore streamed content chunks."""
        pass

    def on_reasoning_chunk(self, chunk: str) -> None:
        """Ignore streamed reasoning chunks."""
        pass

    def on_chunks_end(self) -> None:
        """Ignore the end-of-stream notification."""
        pass


@dataclass(frozen=True)
class Usage:
    tokens: int
    cost: float


@dataclass(frozen=True)
class Completion:
    message: AssistantMessage
    usage: Optional[Usage] = None


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
