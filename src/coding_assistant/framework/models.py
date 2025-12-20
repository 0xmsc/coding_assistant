from dataclasses import dataclass, field
from typing import Literal, Optional, Union


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str  # JSON string


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: FunctionCall


@dataclass(frozen=True)
class SystemMessage:
    content: str
    role: Literal["system"] = "system"
    name: Optional[str] = None


@dataclass(frozen=True)
class UserMessage:
    content: str
    role: Literal["user"] = "user"
    name: Optional[str] = None


@dataclass(frozen=True)
class AssistantMessage:
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: Literal["assistant"] = "assistant"
    name: Optional[str] = None


@dataclass(frozen=True)
class ToolMessage:
    content: str
    tool_call_id: str
    role: Literal["tool"] = "tool"
    name: Optional[str] = None


LLMMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]
