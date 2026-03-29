from __future__ import annotations

import asyncio
from collections import deque
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
PromptContent = str | list[dict[str, Any]]


@dataclass(frozen=True)
class SessionState:
    """Snapshot of whether the session can accept or is running work."""

    promptable: bool
    running: bool
    queued_prompt_count: int


@dataclass(frozen=True)
class StateChangedEvent:
    """Event emitted whenever the session state snapshot changes."""

    state: SessionState


@dataclass(frozen=True)
class ToolCallsEvent:
    """Event emitted when the current run pauses for tool execution."""

    message: AssistantMessage


@dataclass(frozen=True)
class RunFinishedEvent:
    """Event emitted after a completed run is committed to history."""

    summary: str


@dataclass(frozen=True)
class RunCancelledEvent:
    """The current run was cancelled before reaching the next user boundary."""


@dataclass(frozen=True)
class RunFailedEvent:
    """Event emitted when a run fails with an exception."""

    error: str


AgentSessionEvent = (
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


@dataclass(frozen=True)
class _QueuedPrompt:
    content: PromptContent


@dataclass(frozen=True)
class _RunResult:
    history: list[BaseMessage] | None
    summary: str = ""


class AgentSession:
    """Own queued prompts, transcript state, and one concrete agent run loop."""

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
        self._pending_prompts: deque[_QueuedPrompt] = deque()
        self._current_run_task: asyncio.Task[_RunResult] | None = None
        self._subscribers: list[asyncio.Queue[AgentSessionEvent]] = []
        self._mutation_lock = asyncio.Lock()
        self._run_loop_wakeup = asyncio.Event()
        self._closed = False
        self._run_loop_task = asyncio.create_task(self._run_loop())

    @property
    def state(self) -> SessionState:
        """Return the current promptability, run state, and queued prompt count."""
        return SessionState(
            promptable=not self._closed,
            running=self._current_run_task is not None,
            queued_prompt_count=len(self._pending_prompts),
        )

    @property
    def history(self) -> list[BaseMessage]:
        """Return a copy of the committed transcript history."""
        return list(self._history)

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[AgentSessionEvent]]:
        """Subscribe to streamed session events and receive an initial state snapshot."""
        return self._subscribe()

    async def enqueue_prompt(self, content: PromptContent, *, priority: bool = False) -> bool:
        """Queue one prompt, optionally ahead of already queued work."""
        async with self._mutation_lock:
            if self._closed:
                return False
            queued_prompt = _QueuedPrompt(content=content)
            if priority:
                self._pending_prompts.appendleft(queued_prompt)
            else:
                self._pending_prompts.append(queued_prompt)
            self._run_loop_wakeup.set()
        await self._publish_state()
        return True

    async def interrupt_and_enqueue(self, content: PromptContent) -> bool:
        """Cancel the active run and make this prompt the next prompt consumed."""
        async with self._mutation_lock:
            if self._closed:
                return False
            self._pending_prompts.appendleft(_QueuedPrompt(content=content))
            run_task = self._current_run_task
            self._run_loop_wakeup.set()
        await self._publish_state()
        if run_task is not None:
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task
        return True

    async def cancel_current_run(self, *, discard_pending_prompts: bool = False) -> bool:
        """Cancel the active run, optionally clearing prompts that have not started yet."""
        async with self._mutation_lock:
            run_task = self._current_run_task
            had_pending_prompts = bool(self._pending_prompts)
            if discard_pending_prompts:
                self._pending_prompts.clear()

        if run_task is None:
            if discard_pending_prompts and had_pending_prompts:
                await self._publish_state()
                return True
            return False

        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task
        return True

    async def close(self) -> None:
        """Stop the session loop and cancel any active or queued work."""
        async with self._mutation_lock:
            if self._closed:
                return
            self._closed = True
            self._pending_prompts.clear()
            run_task = self._current_run_task
            self._run_loop_wakeup.set()

        if run_task is not None:
            run_task.cancel()
            with suppress(asyncio.CancelledError):
                await run_task

        self._run_loop_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._run_loop_task

    @asynccontextmanager
    async def _subscribe(self) -> AsyncIterator[asyncio.Queue[AgentSessionEvent]]:
        queue: asyncio.Queue[AgentSessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(StateChangedEvent(state=self.state))
        try:
            yield queue
        finally:
            with suppress(ValueError):
                self._subscribers.remove(queue)

    async def _run_loop(self) -> None:
        while True:
            await self._run_loop_wakeup.wait()

            while True:
                async with self._mutation_lock:
                    if self._closed:
                        return
                    if self._current_run_task is not None:
                        break
                    if not self._pending_prompts:
                        self._run_loop_wakeup.clear()
                        break

                    prompt = self._pending_prompts.popleft()
                    self._history.append(UserMessage(content=prompt.content))
                    current_history = list(self._history)
                    run_task = asyncio.create_task(self._run_until_boundary(current_history))
                    self._current_run_task = run_task

                await self._publish_state()
                try:
                    result = await run_task
                except asyncio.CancelledError:
                    result = None
                finally:
                    async with self._mutation_lock:
                        if self._current_run_task is run_task:
                            self._current_run_task = None

                if result is not None and result.history is not None:
                    self._history = result.history
                    self._publish_event(RunFinishedEvent(summary=result.summary))
                await self._publish_state()

    async def _run_until_boundary(self, history: list[BaseMessage]) -> _RunResult:
        current_history = list(history)
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
                        continue
                    self._publish_event(event)

                if boundary is None:
                    raise RuntimeError("run_agent_event_stream stopped without yielding a boundary.")

                if isinstance(boundary, AwaitingToolCalls):
                    self._publish_event(ToolCallsEvent(message=boundary.message))
                    current_history = await execute_tool_calls(
                        boundary=boundary,
                        tools=self._tools,
                    )
                    continue

                current_history = list(boundary.history)
                return _RunResult(
                    history=current_history,
                    summary=_get_latest_assistant_summary(current_history),
                )
        except asyncio.CancelledError:
            self._publish_event(RunCancelledEvent())
            raise
        except Exception as exc:
            self._publish_event(RunFailedEvent(error=str(exc)))
            return _RunResult(history=None)

    async def _publish_state(self) -> None:
        self._publish_event(StateChangedEvent(state=self.state))

    def _publish_event(self, event: AgentSessionEvent) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(event)


def _get_latest_assistant_summary(history: Sequence[BaseMessage]) -> str:
    """Return the newest non-empty assistant text from committed history."""
    for message in reversed(history):
        if isinstance(message, AssistantMessage) and message.content:
            return message.content
    return ""
