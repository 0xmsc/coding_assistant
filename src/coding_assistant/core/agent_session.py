from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from coding_assistant.core.agent import run_agent_event_stream
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.tool_calls import ToolCallLifecycleEvent, ToolExecutionCancelled, execute_tool_calls
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

    running: bool
    queued_prompt_count: int
    paused: bool = False
    pending_prompts: tuple[PromptContent, ...] = ()


@dataclass(frozen=True)
class StateChangedEvent:
    """Event emitted whenever the session state snapshot changes."""

    state: SessionState


@dataclass(frozen=True)
class PromptStartedEvent:
    """Event emitted when a queued prompt begins running."""

    content: PromptContent
    source: str = "local"


@dataclass(frozen=True)
class ToolCallsEvent:
    """Event emitted when the current run pauses for tool execution."""

    message: AssistantMessage
    source: str = "local"


@dataclass(frozen=True)
class ToolCallUpdateEvent:
    """One lifecycle update for a tool call executing inside the current run."""

    event: ToolCallLifecycleEvent
    source: str = "local"


@dataclass(frozen=True)
class RunFinishedEvent:
    """Event emitted after a completed run is committed to history."""

    summary: str
    source: str = "local"


@dataclass(frozen=True)
class RunCancelledEvent:
    """The current run was cancelled before reaching the next user boundary."""

    source: str = "local"


@dataclass(frozen=True)
class RunFailedEvent:
    """Event emitted when a run fails with an exception."""

    error: str
    source: str = "local"


AgentSessionEvent = (
    ContentDeltaEvent
    | ReasoningDeltaEvent
    | StatusEvent
    | CompletionEvent
    | StateChangedEvent
    | PromptStartedEvent
    | ToolCallsEvent
    | ToolCallUpdateEvent
    | RunFinishedEvent
    | RunCancelledEvent
    | RunFailedEvent
)


@dataclass(frozen=True)
class _QueuedPrompt:
    """Internal representation of a prompt waiting to be processed."""

    content: PromptContent
    source: str = "local"


@dataclass(frozen=True)
class _RunResult:
    """Result of a single agent run, returned by `_run_until_boundary`."""

    history: list[BaseMessage] | None  # Updated transcript, or None on fatal error
    source: str  # Origin of the prompt that triggered this run
    summary: str = ""  # Short text summarizing the outcome
    cancelled: bool = False  # True if the run was cancelled
    error: str | None = None  # Error message if the run failed


class AgentSession:
    """Own queued prompts, transcript state, and one concrete agent run loop.

    This session manages two prompt queues:

    1. _pending_prompts: The main queue. Prompts run FIFO when the session is idle.

    2. _pending_steering_prompts: Steers the agent mid-run. When the agent reaches
       any boundary (tool call or user input), steering prompts are injected before
       continuing. This allows redirecting the agent mid-task.

    Session states:
    - Running: actively processing a prompt
    - Idle: not running, waiting for wakeup events (new prompts, cancellation, resume)
    - Paused: not running, pending prompts exist but won't process until resume() is called

    After each run completes (or is cancelled/errored), any steering prompts in
    queue are moved back to the front of the main queue.
    """

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
        # Two queues with different roles:
        # - _pending_prompts: main FIFO queue when session is idle.
        # - _pending_steering_prompts: injected at agent boundaries while a run is active.
        self._pending_prompts: deque[_QueuedPrompt] = deque()
        self._pending_steering_prompts: deque[_QueuedPrompt] = deque()
        self._current_run_task: asyncio.Task[_RunResult] | None = None
        self._subscribers: list[asyncio.Queue[AgentSessionEvent]] = []
        # Protects all state mutations; public mutating methods acquire this lock.
        self._mutation_lock = asyncio.Lock()
        self._run_loop_wakeup = asyncio.Event()
        self._closed = False
        self._paused = False
        self._run_loop_task = asyncio.create_task(self._run_loop())

    @property
    def state(self) -> SessionState:
        """Return the current run state and pending prompt count."""
        pending_prompts = tuple(prompt.content for prompt in self._pending_steering_prompts) + tuple(
            prompt.content for prompt in self._pending_prompts
        )
        return SessionState(
            running=self._current_run_task is not None,
            queued_prompt_count=len(pending_prompts),
            paused=self._paused,
            pending_prompts=pending_prompts,
        )

    @property
    def history(self) -> list[BaseMessage]:
        """Return a copy of the committed transcript history."""
        return list(self._history)

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[AgentSessionEvent]]:
        """Subscribe to streamed session events and receive an initial state snapshot."""
        return self._subscribe()

    async def enqueue_prompt(self, content: PromptContent, *, source: str = "local") -> bool:
        """Queue one prompt."""
        async with self._mutation_lock:
            if self._closed:
                return False
            self._paused = False
            self._pending_prompts.append(_QueuedPrompt(content=content, source=source))
            self._run_loop_wakeup.set()
        await self._publish_state()
        return True

    async def enqueue_prompt_if_idle(self, content: PromptContent, *, source: str = "local") -> bool:
        """Queue one prompt only when the session has no running or pending work."""
        async with self._mutation_lock:
            if (
                self._closed
                or self._current_run_task is not None
                or self._paused
                or self._pending_prompts
                or self._pending_steering_prompts
            ):
                return False
            self._paused = False
            self._pending_prompts.append(_QueuedPrompt(content=content, source=source))
            self._run_loop_wakeup.set()
        await self._publish_state()
        return True

    async def enqueue_steering_prompt(self, content: PromptContent, *, source: str = "local") -> bool:
        """Inject one prompt into the active run at the next agent-loop boundary."""
        async with self._mutation_lock:
            if self._closed:
                return False
            self._paused = False
            queued_prompt = _QueuedPrompt(content=content, source=source)
            if self._current_run_task is None:
                self._pending_prompts.appendleft(queued_prompt)
                self._run_loop_wakeup.set()
            else:
                self._pending_steering_prompts.append(queued_prompt)
        await self._publish_state()
        return True

    async def pop_last_queued_prompt(self) -> PromptContent | None:
        """Remove and return the last queued prompt, or None if the queue is empty."""
        async with self._mutation_lock:
            if self._closed:
                return None
            if self._pending_prompts:
                last = self._pending_prompts.pop()
            elif self._pending_steering_prompts:
                last = self._pending_steering_prompts.pop()
            else:
                return None
        await self._publish_state()
        return last.content

    async def cancel_current_run(self, *, discard_pending_prompts: bool = False, pause_queue: bool = False) -> bool:
        """Cancel the active run, optionally clearing prompts that have not started yet."""
        async with self._mutation_lock:
            run_task = self._current_run_task
            had_pending_prompts = bool(self._pending_prompts or self._pending_steering_prompts)
            was_paused = self._paused
            self._paused = pause_queue and had_pending_prompts
            if discard_pending_prompts:
                self._pending_prompts.clear()
                self._pending_steering_prompts.clear()
                self._paused = False

        if run_task is None:
            if (discard_pending_prompts and had_pending_prompts) or self._paused != was_paused:
                await self._publish_state()
                return True
            return False

        if self._paused != was_paused:
            await self._publish_state()
        run_task.cancel()
        with suppress(asyncio.CancelledError):
            await run_task
        return True

    async def resume(self) -> bool:
        """Resume consuming queued prompts after an explicit pause."""
        async with self._mutation_lock:
            if self._closed or not self._paused:
                return False
            self._paused = False
            if self._pending_prompts:
                self._run_loop_wakeup.set()
        await self._publish_state()
        return True

    async def close(self) -> None:
        """Stop the session loop and cancel any active or queued work."""
        async with self._mutation_lock:
            if self._closed:
                return
            self._closed = True
            self._pending_prompts.clear()
            self._pending_steering_prompts.clear()
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
        """Create a new event subscription queue and deliver an initial state snapshot.

        Yields:
            A queue that will receive all subsequent session events.

        Note:
            The queue is automatically removed from subscribers on exit.
        """
        queue: asyncio.Queue[AgentSessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(StateChangedEvent(state=self.state))
        try:
            yield queue
        finally:
            with suppress(ValueError):
                self._subscribers.remove(queue)

    async def _run_loop(self) -> None:
        """Main event loop driving prompt processing.

        Structure:
        - Outer loop: wait for _run_loop_wakeup event.
        - Inner loop: while idle and prompts available, run them sequentially.

        The inner loop exits when:
        - Session is closed (exit outer loop entirely)
        - A run is already active (another task started it)
        - Session is paused (wait for resume())
        - No pending prompts (clear wakeup and return to outer wait)

        After each run completes (success, cancellation, or error), steering prompts
        are drained back to the main queue (LIFO) so they run before new prompts.
        """
        while True:
            await self._run_loop_wakeup.wait()

            while True:
                async with self._mutation_lock:
                    if self._closed:
                        return
                    if self._current_run_task is not None:
                        break  # Another task is running
                    if self._paused:
                        self._run_loop_wakeup.clear()
                        break  # Waiting for explicit resume
                    if not self._pending_prompts:
                        self._run_loop_wakeup.clear()
                        break  # Nothing to do, go back to waiting

                    # Start running the next prompt
                    prompt = self._pending_prompts.popleft()
                    current_history = [*self._history, UserMessage(content=prompt.content)]
                    run_task = asyncio.create_task(self._run_until_boundary(current_history, source=prompt.source))
                    self._current_run_task = run_task

                self._publish_event(PromptStartedEvent(content=prompt.content, source=prompt.source))
                await self._publish_state()
                try:
                    result = await run_task
                except asyncio.CancelledError:
                    result = None
                finally:
                    async with self._mutation_lock:
                        if self._current_run_task is run_task:
                            self._current_run_task = None
                        # On cancellation or error, move steering prompts back to main queue
                        if result is not None and (result.cancelled or result.error is not None):
                            self._drain_steering_prompts_to_front_of_queue()

                if result is not None and result.history is not None:
                    self._history = result.history

                if result is not None and result.cancelled:
                    self._publish_event(RunCancelledEvent(source=result.source))
                elif result is not None and result.error is not None:
                    self._publish_event(RunFailedEvent(error=result.error, source=result.source))
                elif result is not None and result.history is not None:
                    self._publish_event(RunFinishedEvent(summary=result.summary, source=result.source))
                await self._publish_state()

    async def _run_until_boundary(self, history: list[BaseMessage], *, source: str) -> _RunResult:
        """Run the agent from the given history until the next user boundary.

        The agent stream is consumed until it yields either:
        - AwaitingUser: the assistant has produced a message and is waiting for user input.
        - AwaitingToolCalls: the assistant has requested tool calls; these are executed.

        After handling tool calls, the loop continues until the next boundary.
        Steering prompts are injected at each boundary if available.

        Args:
            history: The starting conversation history (includes the initial user prompt).
            source: Identification string for the origin of this run (for event tracing).

        Returns:
            A `_RunResult` capturing the final history, outcome, and optional error.
        """
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
                    self._publish_event(ToolCallsEvent(message=boundary.message, source=source))
                    current_history = await execute_tool_calls(
                        boundary=boundary,
                        tools=self._tools,
                        on_event=lambda event: self._publish_event(ToolCallUpdateEvent(event=event, source=source)),
                    )
                    steering_prompt = await self._pop_next_steering_prompt()
                    if steering_prompt is not None:
                        current_history.append(UserMessage(content=steering_prompt.content))
                        self._publish_event(
                            PromptStartedEvent(content=steering_prompt.content, source=steering_prompt.source),
                        )
                        await self._publish_state()
                    continue

                steering_prompt = await self._pop_next_steering_prompt()
                if steering_prompt is not None:
                    current_history = [*boundary.history, UserMessage(content=steering_prompt.content)]
                    self._publish_event(
                        PromptStartedEvent(content=steering_prompt.content, source=steering_prompt.source),
                    )
                    await self._publish_state()
                    continue

                current_history = list(boundary.history)
                return _RunResult(
                    history=current_history,
                    source=source,
                    summary=_get_latest_assistant_summary(current_history),
                )
        except ToolExecutionCancelled as exc:
            return _RunResult(history=exc.history, source=source, cancelled=True)
        except asyncio.CancelledError:
            return _RunResult(history=current_history, source=source, cancelled=True)
        except Exception as exc:
            return _RunResult(history=None, source=source, error=str(exc))

    async def _pop_next_steering_prompt(self) -> _QueuedPrompt | None:
        """Remove and return the next steering prompt, if any.

        Acquires the mutation lock and respects the closed state.

        Returns:
            The next steering prompt, or None if none available or session closed.
        """
        async with self._mutation_lock:
            if self._closed or not self._pending_steering_prompts:
                return None
            return self._pending_steering_prompts.popleft()

    def _drain_steering_prompts_to_front_of_queue(self) -> None:
        """Move all steering prompts back to the front of the main queue.

        Called after a run ends (completion, cancellation, or error) so that
        any steering prompts that were injected mid-run get re-queued as regular
        prompts and processed in LIFO order before new prompts.
        """
        while self._pending_steering_prompts:
            self._pending_prompts.appendleft(self._pending_steering_prompts.pop())

    async def _publish_state(self) -> None:
        """Publish a state snapshot event to all subscribers."""
        self._publish_event(StateChangedEvent(state=self.state))

    def _publish_event(self, event: AgentSessionEvent) -> None:
        """Broadcast an event to all subscriber queues.

        Uses `list(self._subscribers)` to allow safe mutation during iteration
        (e.g., subscribers unsubscribing in response to events).
        """
        for queue in list(self._subscribers):
            queue.put_nowait(event)


def _get_latest_assistant_summary(history: Sequence[BaseMessage]) -> str:
    """Return the newest non-empty assistant text from committed history."""
    for message in reversed(history):
        if isinstance(message, AssistantMessage) and message.content:
            return message.content
    return ""
