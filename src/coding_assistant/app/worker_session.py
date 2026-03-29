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
    promptable: bool
    running: bool


@dataclass(frozen=True)
class StateChangedEvent:
    state: WorkerState


@dataclass(frozen=True)
class ToolCallsEvent:
    message: AssistantMessage


@dataclass(frozen=True)
class RunFinishedEvent:
    summary: str


@dataclass(frozen=True)
class RunCancelledEvent:
    """The current run was cancelled."""


@dataclass(frozen=True)
class RunFailedEvent:
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
        return WorkerState(
            promptable=self._run_task is None,
            running=self._run_task is not None,
        )

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[WorkerSessionEvent]]:
        return self._subscribe()

    async def submit_prompt(self, content: str | list[dict[str, Any]]) -> bool:
        if self._run_task is not None:
            return False

        self._history.append(UserMessage(content=content))
        self._run_task = asyncio.create_task(self._run_until_boundary())
        await self._publish_state()
        return True

    async def cancel_current_run(self) -> bool:
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
        queue: asyncio.Queue[WorkerSessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(StateChangedEvent(state=self.state))
        try:
            yield queue
        finally:
            with suppress(ValueError):
                self._subscribers.remove(queue)

    async def _run_until_boundary(self) -> None:
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
        self._publish_event(StateChangedEvent(state=self.state))

    def _publish_event(self, event: WorkerSessionEvent) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(event)


def _get_latest_assistant_summary(history: Sequence[BaseMessage]) -> str:
    for message in reversed(history):
        if isinstance(message, AssistantMessage) and message.content:
            return message.content
    return ""
