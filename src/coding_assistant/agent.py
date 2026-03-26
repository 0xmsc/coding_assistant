from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import (
    BaseMessage,
    ContentDeltaEvent as LLMContentDeltaEvent,
    CompletionEvent,
    ReasoningDeltaEvent as LLMReasoningDeltaEvent,
    StatusEvent as LLMStatusEvent,
    Tool,
)
from coding_assistant.llm.types import AssistantMessage, StatusLevel


@dataclass(frozen=True)
class ContentDeltaEvent:
    """One streamed assistant content chunk from the agent."""

    content: str


@dataclass(frozen=True)
class ReasoningDeltaEvent:
    """One streamed reasoning chunk from the agent."""

    content: str


@dataclass(frozen=True)
class StatusEvent:
    """One non-content runtime status update."""

    message: str
    level: StatusLevel = StatusLevel.INFO


@dataclass(frozen=True)
class AwaitingToolsEvent:
    """The assistant produced tool calls and is waiting for tool results."""

    message: AssistantMessage


@dataclass(frozen=True)
class AwaitingUserEvent:
    """The assistant yielded to the caller without requesting tools."""

    message: AssistantMessage | None = None


@dataclass(frozen=True)
class FailedEvent:
    """The run failed before reaching the next external boundary."""

    error: str


AgentEvent: TypeAlias = (
    ContentDeltaEvent | ReasoningDeltaEvent | StatusEvent | AwaitingToolsEvent | AwaitingUserEvent | FailedEvent
)


def _validate_tools(tools: Sequence[Tool]) -> list[Tool]:
    """Reject duplicate tool names before invoking the model."""
    validated_tools = list(tools)
    names: set[str] = set()
    for tool in validated_tools:
        name = tool.name()
        if name in names:
            raise ValueError(f"Duplicate tool name: {name}")
        names.add(name)

    return validated_tools


def _should_wait_for_user(history: Sequence[BaseMessage]) -> bool:
    """Return true when the transcript already ends in assistant-owned output."""
    return history[-1].role not in {"user", "tool"}


async def run_agent(
    *,
    history: Sequence[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    streamer: Any = openai_stream_completion,
) -> AsyncIterator[AgentEvent]:
    """Advance the transcript until the next external boundary and stream events."""
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent requires a non-empty history.")

    if _should_wait_for_user(current_history):
        yield AwaitingUserEvent()
        return

    all_tools = _validate_tools(tools)

    try:
        async for event in streamer(
            current_history,
            model=model,
            tools=all_tools,
        ):
            if isinstance(event, LLMContentDeltaEvent):
                yield ContentDeltaEvent(content=event.content)
            elif isinstance(event, LLMReasoningDeltaEvent):
                yield ReasoningDeltaEvent(content=event.content)
            elif isinstance(event, LLMStatusEvent):
                yield StatusEvent(message=event.message, level=event.level)
            elif isinstance(event, CompletionEvent):
                message = event.completion.message
                if message.tool_calls:
                    yield AwaitingToolsEvent(message=message)
                else:
                    yield AwaitingUserEvent(message=message)
                return
    except Exception as exc:
        error = str(exc) or exc.__class__.__name__
        yield FailedEvent(error=error)
