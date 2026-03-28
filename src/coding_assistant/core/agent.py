from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from coding_assistant.core.boundaries import (
    AgentBoundary,
    AwaitingToolCalls,
    AwaitingUser,
    get_pending_tool_call_message,
)
from coding_assistant.core.tool_calls import build_tools, execute_tool_calls
from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompletionEvent,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    Tool,
)


def _should_wait_for_user(history: Sequence[BaseMessage]) -> bool:
    """Return true when the transcript already ends in assistant-owned output."""
    return history[-1].role not in {"user", "tool"}


async def run_agent(
    *,
    history: Sequence[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    streamer: Any = openai_stream_completion,
) -> list[BaseMessage]:
    """Advance the transcript until the assistant yields back to the caller."""
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent requires a non-empty history.")

    while True:
        boundary: AgentBoundary | None = None
        async for event in run_agent_event_stream(
            history=current_history,
            model=model,
            tools=tools,
            streamer=streamer,
        ):
            if isinstance(event, (AwaitingUser, AwaitingToolCalls)):
                boundary = event

        if boundary is None:
            raise RuntimeError("run_agent_event_stream stopped without yielding a boundary.")

        if isinstance(boundary, AwaitingUser):
            return boundary.history

        current_history = await execute_tool_calls(
            boundary=boundary,
            tools=tools,
        )


def _get_boundary(history: list[BaseMessage]) -> AgentBoundary | None:
    """Return the current caller-owned boundary when one already exists."""
    if get_pending_tool_call_message(history) is not None:
        return AwaitingToolCalls(history=history)
    if _should_wait_for_user(history):
        return AwaitingUser(history=history)
    return None


async def run_agent_event_stream(
    *,
    history: Sequence[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    streamer: Any = openai_stream_completion,
) -> AsyncIterator[
    ContentDeltaEvent | ReasoningDeltaEvent | StatusEvent | CompletionEvent | AwaitingUser | AwaitingToolCalls
]:
    """Yield streamed LLM events and end with a terminal boundary object."""
    current_history = list(history)
    if not current_history:
        raise ValueError("run_agent_event_stream requires a non-empty history.")

    immediate_boundary = _get_boundary(current_history)
    if immediate_boundary is not None:
        yield immediate_boundary
        return

    all_tools = build_tools(tools=tools)
    completion_message: AssistantMessage | None = None
    async for event in streamer(
        current_history,
        model=model,
        tools=all_tools,
    ):
        yield event
        if isinstance(event, CompletionEvent):
            completion_message = event.completion.message

    if completion_message is None:
        raise RuntimeError("Streamer stopped without yielding a completion.")

    current_history.append(completion_message)
    boundary = _get_boundary(current_history)
    if boundary is None:
        raise RuntimeError("run_agent_event_stream did not reach a boundary after the completion.")
    yield boundary
