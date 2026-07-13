from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any

from coding_assistant.core.tool_calls import (
    ToolCallExecutionCompleted,
    ToolExecutionCancelled,
    ToolMessageProduced,
    build_tools,
    stream_tool_call_execution,
)
from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompletionEvent,
    ContentDeltaEvent,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    Tool,
    Usage,
    UserMessage,
)


logger = logging.getLogger(__name__)


CompletionStreamer = Callable[[Sequence[BaseMessage], Sequence[Tool], str], AsyncIterator[object]]
PromptContent = str | list[dict[str, Any]]


@dataclass(frozen=True)
class SessionState:
    """Snapshot of whether the session can accept or is running work."""

    running: bool
    paused: bool = False
    pending_prompts: tuple[PromptContent, ...] = ()
    usage: Usage | None = None


@dataclass(frozen=True)
class PromptStartedEvent:
    """Event emitted when a queued prompt begins running."""

    content: PromptContent


@dataclass(frozen=True)
class RunFinishedEvent:
    """Event emitted after a completed run is committed to history."""

    summary: str


@dataclass(frozen=True)
class RunCancelledEvent:
    """The current run was cancelled before reaching the next user boundary."""

    pass


@dataclass(frozen=True)
class RunFailedEvent:
    """Event emitted when a run fails with an exception."""

    error: str


AgentSessionEvent = (
    ContentDeltaEvent
    | ReasoningDeltaEvent
    | ModelRetryEvent
    | StatusEvent
    | CompletionEvent
    | PromptStartedEvent
    | ToolMessageProduced
    | RunFinishedEvent
    | RunCancelledEvent
    | RunFailedEvent
)


@dataclass(frozen=True)
class _QueuedPrompt:
    """Internal representation of a prompt waiting to be processed."""

    content: PromptContent


@dataclass(frozen=True)
class _RunResult:
    """Result of a single queued prompt run."""

    history: list[BaseMessage] | None  # Updated transcript, or None on fatal error
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
        self._tools = build_tools(tools=tools)
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
        self._usage = Usage(tokens=0, cost=0.0)
        self._run_loop_task = asyncio.create_task(self._run_loop())

    @property
    def state(self) -> SessionState:
        """Return the current run state and pending prompt count."""
        pending_prompts = tuple(prompt.content for prompt in self._pending_steering_prompts) + tuple(
            prompt.content for prompt in self._pending_prompts
        )
        return SessionState(
            running=self._current_run_task is not None,
            paused=self._paused,
            pending_prompts=pending_prompts,
            usage=self._usage,
        )

    @property
    def history(self) -> list[BaseMessage]:
        """Return a copy of the committed transcript history."""
        return list(self._history)

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[AgentSessionEvent]]:
        """Subscribe to streamed session events."""
        return self._subscribe()

    async def enqueue_prompt(self, content: PromptContent) -> bool:
        """Queue one prompt."""
        async with self._mutation_lock:
            if self._closed:
                return False
            self._paused = False
            self._pending_prompts.append(_QueuedPrompt(content=content))
            self._run_loop_wakeup.set()
        return True

    async def enqueue_prompt_if_idle(self, content: PromptContent) -> bool:
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
            self._pending_prompts.append(_QueuedPrompt(content=content))
            self._run_loop_wakeup.set()
        return True

    async def enqueue_steering_prompt(self, content: PromptContent) -> bool:
        """Inject one prompt into the active run at the next agent-loop boundary."""
        async with self._mutation_lock:
            if self._closed:
                return False
            self._paused = False
            queued_prompt = _QueuedPrompt(content=content)
            if self._current_run_task is None:
                self._pending_prompts.appendleft(queued_prompt)
                self._run_loop_wakeup.set()
            else:
                self._pending_steering_prompts.append(queued_prompt)
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
        return last.content

    async def cancel_current_run(self, *, pause_queue: bool = False) -> bool:
        """Cancel the active run, optionally pausing the queue."""
        async with self._mutation_lock:
            run_task = self._current_run_task
            had_pending_prompts = bool(self._pending_prompts or self._pending_steering_prompts)
            was_paused = self._paused
            self._paused = pause_queue and had_pending_prompts

        if run_task is None:
            return self._paused != was_paused

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
        """Create a new event subscription queue.

        Yields:
            A queue that will receive all subsequent session events.

        Note:
            The queue is automatically removed from subscribers on exit.
        """
        queue: asyncio.Queue[AgentSessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
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
                    run_task = asyncio.create_task(self._run_prompt(current_history))
                    self._current_run_task = run_task

                self._publish_event(PromptStartedEvent(content=prompt.content))
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
                    self._publish_event(RunCancelledEvent())
                elif result is not None and result.error is not None:
                    self._publish_event(RunFailedEvent(error=result.error))
                elif result is not None and result.history is not None:
                    self._publish_event(RunFinishedEvent(summary=result.summary))

    async def _run_prompt(self, history: list[BaseMessage]) -> _RunResult:
        """Run model and tool turns until the agent needs another user prompt.

        Args:
            history: The starting conversation history (includes the initial user prompt).
        Returns:
            A `_RunResult` capturing the final history, outcome, and optional error.
        """
        current_history = list(history)
        streamer = self._completion_streamer or openai_stream_completion
        try:
            while True:
                completion_message: AssistantMessage | None = None
                async for event in streamer(current_history, self._tools, self._model):
                    if not isinstance(
                        event,
                        (ContentDeltaEvent, ReasoningDeltaEvent, ModelRetryEvent, StatusEvent, CompletionEvent),
                    ):
                        raise TypeError(f"Completion streamer yielded unsupported event: {type(event).__name__}.")
                    if isinstance(event, CompletionEvent):
                        completion_message = event.completion.message
                        if event.completion.usage is not None:
                            async with self._mutation_lock:
                                cost = event.completion.usage.cost
                                if cost is not None and self._usage.cost is not None:
                                    cost += self._usage.cost
                                else:
                                    cost = None
                                self._usage = Usage(
                                    tokens=event.completion.usage.tokens,
                                    cost=cost,
                                )
                        self._publish_event(event)
                        continue
                    self._publish_event(event)

                if completion_message is None:
                    raise RuntimeError("Completion streamer stopped without yielding a completion.")
                current_history.append(completion_message)

                if completion_message.tool_calls:
                    completed_history: list[BaseMessage] | None = None
                    async for item in stream_tool_call_execution(
                        history=current_history,
                        message=completion_message,
                        tools=self._tools,
                    ):
                        if isinstance(item, ToolMessageProduced):
                            self._publish_event(item)
                            continue
                        if isinstance(item, ToolCallExecutionCompleted):
                            completed_history = item.history

                    if completed_history is None:
                        raise RuntimeError("Tool execution stopped without returning history.")
                    current_history = completed_history

                steering_prompt = await self._pop_next_steering_prompt()
                if steering_prompt is not None:
                    current_history.append(UserMessage(content=steering_prompt.content))
                    self._publish_event(PromptStartedEvent(content=steering_prompt.content))
                    continue

                if completion_message.tool_calls:
                    continue

                return _RunResult(
                    history=current_history,
                    summary=_get_latest_assistant_summary(current_history),
                )
        except ToolExecutionCancelled as exc:
            return _RunResult(history=exc.history, cancelled=True)
        except asyncio.CancelledError:
            return _RunResult(history=current_history, cancelled=True)
        except Exception as exc:
            logger.exception("Agent session run failed.")
            return _RunResult(history=None, error=str(exc))

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
