from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from coding_assistant.llm.types import ToolResult as LLMToolResult


class ToolResult(LLMToolResult, ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


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
