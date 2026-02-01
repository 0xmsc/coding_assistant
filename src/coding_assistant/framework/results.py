from dataclasses import dataclass
from typing import Any
from coding_assistant.llm.types import ToolResult


@dataclass
class TextResult(ToolResult):
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content}


@dataclass
class FinishTaskResult(ToolResult):
    """Signals that the agent's task is complete."""

    result: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"result": self.result, "summary": self.summary}


@dataclass
class CompactConversationResult(ToolResult):
    """Signals that the conversation history should be summarized."""

    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary}


def result_from_dict(data: dict[str, Any]) -> ToolResult:
    """Reconstruct a ToolResult from its dictionary representation."""
    if "result" in data and "summary" in data:
        return FinishTaskResult(result=data["result"], summary=data["summary"])
    if "summary" in data:
        return CompactConversationResult(summary=data["summary"])
    if "content" in data:
        return TextResult(content=data["content"])

    # Fallback to TextResult with the raw data if we can't determine the type
    return TextResult(content=str(data))
