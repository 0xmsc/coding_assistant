from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Protocol, Awaitable

from coding_assistant.agents.parameters import Parameter
from coding_assistant.tools.mcp import MCPServer
from coding_assistant.agents.callbacks import AgentCallbacks
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


@dataclass
class AgentOutput:
    result: str
    summary: str


@dataclass
class Agent:
    name: str
    model: str

    description: str
    parameters: list[Parameter]

    # This is a function that can validate an agents output.
    # If it returns a string, it will be given to the agent as feedback.
    feedback_function: Callable

    tools: list[Tool]
    mcp_servers: list[MCPServer]
    tool_confirmation_patterns: list[str]

    history: list = field(default_factory=list)
    output: AgentOutput | None = None


class Completer(Protocol):
    """Async callable that produces a model completion.

    Contract:
    - inputs: conversation messages, model name, tool definitions, and callbacks
    - output: Completion(message, tokens)
    """

    def __call__(
        self,
        messages: list[dict],
        *,
        model: str,
        tools: list,
        callbacks: AgentCallbacks,
    ) -> Awaitable[Completion]:
        ...
