from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from coding_assistant.agents.parameters import Parameter
from coding_assistant.tools.mcp import MCPServer


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
    feedback: str | None = None


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
    feedback: str | None


@dataclass
class Agent:
    name: str
    model: str

    description: str
    parameters: list[Parameter]

    # This is a function that can validate an agents output.
    # If it returns a string, it will be given to the agent as feedback.
    feedback_function: Callable

    tools: list[Tool] = field(default_factory=list)
    mcp_servers: list[MCPServer] = field(default_factory=list)

    history: list = field(default_factory=list)
    output: AgentOutput | None = None
