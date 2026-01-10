from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Awaitable, Protocol

from coding_assistant.llm.types import (
    BaseMessage,
    Completion,
    Tool,
    ProgressCallbacks as LLMProgressCallbacks,
)
from coding_assistant.framework.parameters import Parameter


@dataclass(frozen=True)
class AgentDescription:
    name: str
    model: str
    parameters: list[Parameter]
    tools: list[Tool]


@dataclass
class AgentOutput:
    result: str
    summary: str


@dataclass
class AgentState:
    history: list[BaseMessage] = field(default_factory=list)
    output: AgentOutput | None = None


@dataclass
class AgentContext:
    desc: AgentDescription
    state: AgentState


class Completer(Protocol):
    def __call__(
        self,
        messages: list[BaseMessage],
        *,
        model: str,
        tools: Sequence[Tool],
        callbacks: LLMProgressCallbacks,
    ) -> Awaitable[Completion]: ...
