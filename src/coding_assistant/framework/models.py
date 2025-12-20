from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional


from dacite import from_dict


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str  # JSON string


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: FunctionCall
    type: str = "function"


@dataclass(frozen=True, kw_only=True)
class LLMMessage:
    role: str
    content: Optional[str | list[dict]] = None
    name: Optional[str] = None
    provider_specific_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class SystemMessage(LLMMessage):
    content: str | list[dict]
    role: Literal["system"] = "system"


@dataclass(frozen=True, kw_only=True)
class UserMessage(LLMMessage):
    content: str | list[dict]
    role: Literal["user"] = "user"


@dataclass(frozen=True, kw_only=True)
class AssistantMessage(LLMMessage):
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: Literal["assistant"] = "assistant"


@dataclass(frozen=True, kw_only=True)
class ToolMessage(LLMMessage):
    content: str
    tool_call_id: str
    role: Literal["tool"] = "tool"


def message_from_dict(d: dict[str, Any]) -> LLMMessage:
    # Ensure role is set for the Literal types in the dataclasses
    role = d["role"]

    # Filter out None values to avoid dacite WrongTypeError for fields with defaults
    # or Literals.
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


def message_to_dict(msg: LLMMessage) -> dict[str, Any]:
    def factory(data):
        return {k: v for k, v in data if v is not None and v != [] and v != {}}

    return asdict(msg, dict_factory=factory)
