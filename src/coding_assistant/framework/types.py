from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Protocol

from coding_assistant.framework.callbacks import ProgressCallbacks
from coding_assistant.llm.types import LLMMessage, Completion, Tool as LLMTool, ToolResult as LLMToolResult
from coding_assistant.framework.parameters import Parameter


class ToolResult(LLMToolResult, ABC):
    """Base class for all tool results."""

    @abstractmethod
    def to_dict(self) -> dict: ...


@dataclass
class TextResult(ToolResult):
    """Represents a simple text result from a tool."""

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


class Tool(LLMTool, ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def parameters(self) -> dict: ...

    @abstractmethod
    async def execute(self, parameters) -> ToolResult: ...


# Immutable description of an agent
@dataclass(frozen=True)
class AgentDescription:
    name: str
    model: str
    parameters: list[Parameter]
    tools: list[Tool]


# Final output of an agent run
@dataclass
class AgentOutput:
    result: str
    summary: str


# Mutable state for an agent's execution
@dataclass
class AgentState:
    history: list[LLMMessage] = field(default_factory=list)
    output: AgentOutput | None = None


# Combines the immutable description with the mutable state of an agent
@dataclass
class AgentContext:
    desc: AgentDescription
    state: AgentState


class Completer(Protocol):
    def __call__(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        tools: list,
        callbacks: ProgressCallbacks,
    ) -> Awaitable[Completion]: ...
