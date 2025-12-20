from dataclasses import dataclass, field, fields
from typing import Any, Literal, Optional, Union


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str  # JSON string


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: FunctionCall


@dataclass(frozen=True)
class MessageBase:
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": getattr(self, "role")}
        for f in fields(self):
            if f.name == "role":
                continue
            val = getattr(self, f.name)
            if val:  # Only include if truthy (handles None, [], "")
                if f.name == "tool_calls":
                    d[f.name] = [
                        {
                            "id": tc.id,
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in val
                    ]
                else:
                    d[f.name] = val
        return d

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)


@dataclass(frozen=True)
class SystemMessage(MessageBase):
    content: str
    role: Literal["system"] = "system"
    name: Optional[str] = None


@dataclass(frozen=True)
class UserMessage(MessageBase):
    content: str
    role: Literal["user"] = "user"
    name: Optional[str] = None


@dataclass(frozen=True)
class AssistantMessage(MessageBase):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: Literal["assistant"] = "assistant"
    name: Optional[str] = None


@dataclass(frozen=True)
class ToolMessage(MessageBase):
    content: str
    tool_call_id: str
    role: Literal["tool"] = "tool"
    name: Optional[str] = None


LLMMessage = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]
