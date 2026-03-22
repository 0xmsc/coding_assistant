from dataclasses import dataclass
from typing import Any

from coding_assistant.llm.types import ToolResult


@dataclass
class TextResult(ToolResult):
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content}


def normalize_tool_result(result: ToolResult | str) -> str:
    if isinstance(result, str):
        return result
    if hasattr(result, "content"):
        content = getattr(result, "content")
        if isinstance(content, str):
            return content
    return f"Tool produced result of type {type(result).__name__}"


@dataclass
class CompactConversationResult(ToolResult):
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary}
