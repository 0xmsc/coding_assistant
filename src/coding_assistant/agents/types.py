from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Protocol

from coding_assistant.agents.callbacks import AgentProgressCallbacks
from coding_assistant.agents.parameters import Parameter
from coding_assistant.llm.model import Completion


class ToolResult(ABC):
    """Base class for all tool results."""

    pass


@dataclass
class TextResult(ToolResult):
    """Represents a simple text result from a tool."""

    content: str


@dataclass
class FinishTaskResult(ToolResult):
    """Signals that the agent's task is complete."""

    result: str
    summary: str


@dataclass
class ShortenConversationResult(ToolResult):
    """Signals that the conversation history should be summarized."""

    summary: str


class Tool(ABC):
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
    history: list = field(default_factory=list)
    output: AgentOutput | None = None


# Combines the immutable description with the mutable state of an agent
@dataclass
class AgentContext:
    desc: AgentDescription
    state: AgentState


class Completer(Protocol):
    def __call__(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list,
        callbacks: AgentProgressCallbacks,
    ) -> Awaitable[Completion]: ...
