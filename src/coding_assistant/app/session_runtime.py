from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from typing import Any

from coding_assistant.app.session_control import (
    PromptSubmissionResult,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    SessionController,
    SessionControllerName,
    SessionEvent,
    SessionOutput,
    SessionState,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.core.agent import run_agent_event_stream
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.tool_calls import execute_tool_calls
from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import AssistantMessage, BaseMessage, Tool, UserMessage

CompletionStreamer = Callable[[Sequence[BaseMessage], Sequence[Tool], str], AsyncIterator[object]]


class SessionRuntime:
    """Own one live session and coordinate controllers and outputs around it."""

    def __init__(
        self,
        *,
        history: Sequence[BaseMessage],
        model: str,
        tools: Sequence[Tool],
        default_controller: SessionControllerName,
        completion_streamer: CompletionStreamer | None = None,
    ) -> None:
        self._history = list(history)
        self._model = model
        self._tools = list(tools)
        self._default_controller = default_controller
        self._completion_streamer = completion_streamer
        self._run_task: asyncio.Task[None] | None = None
        self._active_controller = default_controller
        self._subscribers: list[asyncio.Queue[SessionEvent]] = []
        self._state_condition = asyncio.Condition()
        self._shutdown_requested = asyncio.Event()

    @property
    def state(self) -> SessionState:
        """Return promptability and who currently owns input."""

        return SessionState(
            promptable=self._run_task is None,
            controller=self._active_controller,
            running=self._run_task is not None,
        )

    @property
    def history(self) -> list[BaseMessage]:
        """Return a snapshot of the current conversation history."""

        return list(self._history)

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[SessionEvent]]:
        """Subscribe to streamed runtime events for outputs and waiting controllers."""

        return self._subscribe()

    async def wait_for_controller_change(self, *, controller: SessionControllerName) -> SessionControllerName:
        """Block until input ownership moves away from `controller`."""

        async with self._state_condition:
            await self._state_condition.wait_for(lambda: self._active_controller != controller)
            return self._active_controller

    async def activate_controller(self, controller: SessionControllerName) -> bool:
        """Transfer input ownership to `controller` if it is not already active."""

        if self._active_controller == controller:
            return False

        self._active_controller = controller
        await self._publish_state()
        return True

    async def activate_default_controller(self, *, cancel_current_run: bool = False) -> None:
        """Restore the default controller, optionally cancelling the current run first."""

        if cancel_current_run and self._active_controller != self._default_controller and self._run_task is not None:
            await self.cancel_current_run()

        if self._active_controller == self._default_controller:
            return

        self._active_controller = self._default_controller
        await self._publish_state()

    async def submit_prompt(
        self,
        *,
        controller: SessionControllerName,
        content: str | list[dict[str, Any]],
    ) -> PromptSubmissionResult:
        """Accept a prompt only from the controller that currently owns input."""

        if controller != self._active_controller:
            return PromptSubmissionResult(accepted=False, reason="inactive_controller")
        return await self._submit_prompt(content)

    async def cancel_current_run(self) -> bool:
        """Cancel the active run and publish the resulting state transition."""

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

    def request_shutdown(self) -> None:
        """Request shutdown of the surrounding runtime run loop."""

        self._shutdown_requested.set()

    async def run(
        self,
        *,
        controllers: Sequence[SessionController],
        outputs: Sequence[SessionOutput],
    ) -> None:
        """Run the configured controllers and outputs until shutdown is requested."""

        adapter_tasks: dict[asyncio.Task[Any], str] = {}
        shutdown_task = asyncio.create_task(self._shutdown_requested.wait())
        pending: set[asyncio.Task[Any]] = {shutdown_task}

        try:
            # Start outputs first so they are subscribed before controllers generate activity.
            for output in outputs:
                task = asyncio.create_task(output.run(self))
                adapter_tasks[task] = output.__class__.__name__
                pending.add(task)

            if outputs:
                await asyncio.sleep(0)

            for controller in controllers:
                task = asyncio.create_task(controller.run(self))
                adapter_tasks[task] = controller.__class__.__name__
                pending.add(task)

            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if shutdown_task in done:
                    break

                for task in done:
                    label = adapter_tasks.pop(task, None)
                    if label is None:
                        continue
                    if task.cancelled():
                        if self._shutdown_requested.is_set():
                            continue
                        raise RuntimeError(f"{label} stopped unexpectedly.")
                    exception = task.exception()
                    if exception is not None:
                        raise exception
                    if self._shutdown_requested.is_set():
                        continue
                    raise RuntimeError(f"{label} stopped unexpectedly.")
        finally:
            self._shutdown_requested.set()
            shutdown_task.cancel()
            with suppress(asyncio.CancelledError):
                await shutdown_task
            for task in adapter_tasks:
                task.cancel()
            if adapter_tasks:
                await asyncio.gather(*adapter_tasks, return_exceptions=True)

    @asynccontextmanager
    async def _subscribe(self) -> AsyncIterator[asyncio.Queue[SessionEvent]]:
        """Register one event subscriber and seed it with the current session state."""

        queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(StateChangedEvent(state=self.state))
        try:
            yield queue
        finally:
            with suppress(ValueError):
                self._subscribers.remove(queue)

    async def _submit_prompt(self, content: str | list[dict[str, Any]]) -> PromptSubmissionResult:
        """Append a user message and start a run if the session is idle."""

        if self._run_task is not None:
            return PromptSubmissionResult(accepted=False, reason="not_ready")

        self._history.append(UserMessage(content=content))
        self._run_task = asyncio.create_task(self._run_until_boundary())
        await self._publish_state()
        return PromptSubmissionResult(accepted=True, reason="accepted")

    async def _run_until_boundary(self) -> None:
        """Drive the agent until it needs tool execution or hands control back to the user."""

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
        """Broadcast the latest session state and wake controller waiters."""

        self._publish_event(StateChangedEvent(state=self.state))
        async with self._state_condition:
            self._state_condition.notify_all()

    def _publish_event(self, event: SessionEvent) -> None:
        """Fan one event out to all current subscribers."""

        for queue in list(self._subscribers):
            queue.put_nowait(event)


def _get_latest_assistant_summary(history: Sequence[BaseMessage]) -> str:
    """Return the last non-empty assistant text message for worker summaries."""

    for message in reversed(history):
        if isinstance(message, AssistantMessage) and message.content:
            return message.content
    return ""
