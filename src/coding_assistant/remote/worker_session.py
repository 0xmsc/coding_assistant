from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from coding_assistant.core.agent import run_agent_event_stream
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.tool_calls import execute_tool_calls
from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompletionEvent,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    Tool,
    UserMessage,
)

CompletionStreamer = Callable[[Sequence[BaseMessage], Sequence[Tool], str], AsyncIterator[object]]


@dataclass(frozen=True)
class WorkerState:
    """Snapshot of whether the worker can accept a prompt and whether it is running."""

    promptable: bool
    running: bool


@dataclass(frozen=True)
class StateChangedEvent:
    """Published whenever the worker's promptability or running state changes."""

    state: WorkerState


@dataclass(frozen=True)
class ToolCallsEvent:
    """Published before tool calls are executed so observers can render them."""

    message: AssistantMessage


@dataclass(frozen=True)
class RunFinishedEvent:
    """Published when a remote prompt finishes and the worker becomes idle again."""

    summary: str


@dataclass(frozen=True)
class RunCancelledEvent:
    """The current run was cancelled."""


@dataclass(frozen=True)
class RunFailedEvent:
    """Published when the worker run aborts due to an unexpected exception."""

    error: str


WorkerSessionEvent = (
    ContentDeltaEvent
    | ReasoningDeltaEvent
    | StatusEvent
    | CompletionEvent
    | StateChangedEvent
    | ToolCallsEvent
    | RunFinishedEvent
    | RunCancelledEvent
    | RunFailedEvent
)


class WorkerSession:
    """Own one remote-controlled worker session."""

    def __init__(
        self,
        *,
        history: Sequence[BaseMessage],
        model: str,
        tools: Sequence[Tool],
        completion_streamer: CompletionStreamer | None = None,
    ) -> None:
        self._history = list(history)
        self._model = model
        self._tools = list(tools)
        self._completion_streamer = completion_streamer
        self._run_task: asyncio.Task[None] | None = None
        self._subscribers: list[asyncio.Queue[WorkerSessionEvent]] = []

    @property
    def state(self) -> WorkerState:
        """Return the current public state snapshot for this worker."""
        return WorkerState(
            promptable=self._run_task is None,
            running=self._run_task is not None,
        )

    @property
    def history(self) -> list[BaseMessage]:
        """Return a copy of the worker transcript accumulated so far."""
        return list(self._history)

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[WorkerSessionEvent]]:
        """Create an event subscription that immediately receives the current state."""
        return self._subscribe()

    async def submit_prompt(self, content: str | list[dict[str, Any]]) -> bool:
        """Start a new remote-controlled run if the worker is currently idle."""
        if self._run_task is not None:
            return False

        self._history.append(UserMessage(content=content))
        self._run_task = asyncio.create_task(self._run_until_boundary())
        await self._publish_state()
        return True

    async def cancel_current_run(self) -> bool:
        """Cancel the active run and publish cancellation if one exists."""
        run_task = self._run_task
        if run_task is None:
            return False

        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task
        if self._run_task is run_task:
            self._run_task = None
            self._publish_event(RunCancelledEvent())
            await self._publish_state()
        return True

    @asynccontextmanager
    async def _subscribe(self) -> AsyncIterator[asyncio.Queue[WorkerSessionEvent]]:
        """Register a queue-based subscriber for worker events."""
        queue: asyncio.Queue[WorkerSessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(StateChangedEvent(state=self.state))
        try:
            yield queue
        finally:
            with suppress(ValueError):
                self._subscribers.remove(queue)

    async def _run_until_boundary(self) -> None:
        """Drive the agent until it reaches the next user boundary or fails."""
        current_history = list(self._history)
        try:
            while True:
                boundary: AwaitingUser | AwaitingToolCalls | None = None
                async for event in run_agent_event_stream(
                    history=current_history,
                    model=self._model,
                    tools=self._tools,
                    streamer=self._completion_streamer or openai_stream_completion,
                ):
                    if isinstance(event, (AwaitingUser, AwaitingToolCalls)):
                        boundary = event
                    else:
                        self._publish_event(event)

                if boundary is None:
                    raise RuntimeError("run_agent_event_stream stopped without yielding a boundary.")

                if isinstance(boundary, AwaitingToolCalls):
                    self._publish_event(ToolCallsEvent(message=boundary.message))
                    current_history = await execute_tool_calls(
                        boundary=boundary,
                        tools=self._tools,
                    )
                    self._history = list(current_history)
                    continue

                current_history = list(boundary.history)
                self._history = current_history
                self._publish_event(RunFinishedEvent(summary=_get_latest_assistant_summary(current_history)))
                return
        except asyncio.CancelledError:
            self._publish_event(RunCancelledEvent())
            raise
        except Exception as exc:
            self._publish_event(RunFailedEvent(error=str(exc)))
        finally:
            self._run_task = None
            await self._publish_state()

    async def _publish_state(self) -> None:
        """Broadcast the latest worker state to all subscribers."""
        self._publish_event(StateChangedEvent(state=self.state))

    def _publish_event(self, event: WorkerSessionEvent) -> None:
        """Fan out one event to every live subscriber queue."""
        for queue in list(self._subscribers):
            queue.put_nowait(event)


def _get_latest_assistant_summary(history: Sequence[BaseMessage]) -> str:
    """Extract the latest assistant text to summarize a completed run."""
    for message in reversed(history):
        if isinstance(message, AssistantMessage) and message.content:
            return message.content
    return ""
