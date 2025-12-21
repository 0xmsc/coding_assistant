from abc import ABC, abstractmethod
from dataclasses import dataclass
from coding_assistant.llm.types import ToolResult as LLMToolResult


class ToolResult(LLMToolResult, ABC):
    @abstractmethod
    def to_dict(self) -> dict: ...


@dataclass
class TextResult(ToolResult):
    content: str

    def to_dict(self):
        return {"content": self.content}


@dataclass
class FinishTaskResult(ToolResult):
    """Signals that the agent's task is complete."""

    result: str
    summary: str

    def to_dict(self):
        return {"result": self.result, "summary": self.summary}


@dataclass
class CompactConversationResult(ToolResult):
    """Signals that the conversation history should be summarized."""

    summary: str

    def to_dict(self):
        return {"summary": self.summary}
