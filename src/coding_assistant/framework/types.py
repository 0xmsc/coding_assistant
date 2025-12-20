from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Awaitable, Protocol

from coding_assistant.llm.types import (
    LLMMessage,
    Completion,
    Tool as LLMTool,
    ProgressCallbacks as LLMProgressCallbacks,
)
from coding_assistant.framework.parameters import Parameter
from coding_assistant.framework.results import (
    CompactConversationResult as CompactConversationResult,
    FinishTaskResult as FinishTaskResult,
    TextResult as TextResult,
    ToolResult as ToolResult,
)


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
        tools: Sequence[LLMTool],
        callbacks: LLMProgressCallbacks,
    ) -> Awaitable[Completion]: ...


# Re-export ToolResult types for convenience if needed,
# but better to import from results.py directly.
# However, many files might expect them here.
