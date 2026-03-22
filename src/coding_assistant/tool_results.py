from dataclasses import dataclass
from typing import Any

from coding_assistant.llm.types import ToolResult


@dataclass
class TextResult(ToolResult):
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content}


@dataclass
class CompactConversationResult(ToolResult):
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary}
