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

    def to_dict(self) -> dict[str, Any]:
        def factory(data):
            return {k: v for k, v in data if v is not None and v != [] and v != {}}

        return asdict(self, dict_factory=factory)


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


def message_from_dict(d: Any) -> LLMMessage:
    def get_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Convert to actual dict if it's a litellm.Message or similar object
    if not isinstance(d, dict):
        role = get_val(d, "role")
        content = get_val(d, "content")
        name = get_val(d, "name")
        provider_specific_fields = get_val(d, "provider_specific_fields")
        reasoning_content = get_val(d, "reasoning_content")
        tool_call_id = get_val(d, "tool_call_id")

        tool_calls = []
        raw_tool_calls = get_val(d, "tool_calls", [])
        for tc in raw_tool_calls or []:
            tc_id = get_val(tc, "id")
            func = get_val(tc, "function")
            func_name = get_val(func, "name")
            func_args = get_val(func, "arguments")
            tool_calls.append(
                {
                    "id": tc_id,
                    "function": {"name": func_name, "arguments": func_args},
                    "type": "function",
                }
            )

        d = {
            "role": role,
            "content": content,
            "name": name,
            "provider_specific_fields": provider_specific_fields or {},
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "tool_call_id": tool_call_id,
        }

    # Ensure role is set for the Literal types in the dataclasses
    if "role" not in d or d["role"] is None:
        # We don't have enough info here to be sure, but usually message_from_dict
        # is called on things that should have a role.
        # However, from_dict will fail if the Literal role is None.
        pass

    role = d.get("role") or "assistant"
    d["role"] = role

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
