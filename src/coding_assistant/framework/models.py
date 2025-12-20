from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class FunctionCall:
    name: str
    arguments: str  # JSON string


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: FunctionCall


@dataclass(frozen=True, kw_only=True)
class LLMMessage:
    role: str
    content: Optional[str | list[dict]] = None
    name: Optional[str] = None
    provider_specific_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name:
            d["name"] = self.name
        if self.provider_specific_fields:
            d["provider_specific_fields"] = self.provider_specific_fields
        return d


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

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return d


@dataclass(frozen=True, kw_only=True)
class ToolMessage(LLMMessage):
    content: str
    tool_call_id: str
    role: Literal["tool"] = "tool"

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["tool_call_id"] = self.tool_call_id
        return d


def message_from_dict(d: Any) -> LLMMessage:
    def get_val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    role = get_val(d, "role") or "assistant"
    name = get_val(d, "name")
    content = get_val(d, "content")
    provider_specific_fields = get_val(d, "provider_specific_fields") or {}

    common: dict[str, Any] = {
        "name": name,
        "provider_specific_fields": provider_specific_fields,
    }

    if role == "system":
        return SystemMessage(content=content or "", **common)
    if role == "user":
        return UserMessage(content=content or "", **common)
    if role == "assistant":
        tool_calls = []
        raw_tool_calls = get_val(d, "tool_calls", [])
        for tc in raw_tool_calls or []:
            tc_id = get_val(tc, "id", "")
            func = get_val(tc, "function")
            func_name = get_val(func, "name", "")
            func_args = get_val(func, "arguments", "")

            tool_calls.append(
                ToolCall(
                    id=tc_id,
                    function=FunctionCall(
                        name=func_name,
                        arguments=func_args,
                    ),
                )
            )
        return AssistantMessage(
            content=content,
            reasoning_content=get_val(d, "reasoning_content"),
            tool_calls=tool_calls,
            **common,
        )
    if role == "tool":
        return ToolMessage(
            content=content or "",
            tool_call_id=get_val(d, "tool_call_id", ""),
            **common,
        )
    raise ValueError(f"Unknown role: {role}")
