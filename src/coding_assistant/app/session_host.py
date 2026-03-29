from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.agent import run_agent_event_stream
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


SessionControllerName = Literal["cli", "remote"]
CLI_CONTROLLER: SessionControllerName = "cli"
REMOTE_CONTROLLER: SessionControllerName = "remote"

CompletionStreamer = Callable[
    [Sequence[BaseMessage], Sequence[Tool], str],
    AsyncIterator[object],
]


@dataclass(frozen=True)
class SessionState:
    promptable: bool
    controller: SessionControllerName
    running: bool

    @property
    def remote_connected(self) -> bool:
        return self.controller == REMOTE_CONTROLLER


@dataclass(frozen=True)
class StateChangedEvent:
    state: SessionState


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


SessionEvent = (
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
class PromptSubmissionResult:
    accepted: bool
    reason: Literal["accepted", "inactive_controller", "not_ready"]


class SessionControlSurface(Protocol):
    @property
    def state(self) -> SessionState: ...

    @property
    def history(self) -> list[BaseMessage]: ...

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[SessionEvent]]: ...

    async def wait_for_controller_change(self, *, controller: SessionControllerName) -> SessionControllerName: ...

    async def activate_controller(self, controller: SessionControllerName) -> bool: ...

    async def restore_cli_controller(self) -> None: ...

    async def submit_prompt(
        self,
        *,
        controller: SessionControllerName,
        content: str | list[dict[str, Any]],
    ) -> PromptSubmissionResult: ...

    async def cancel_current_run(self) -> bool: ...


class SessionHost:
    """Own one live conversation session plus its streamed runtime events."""

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
        self._active_controller = CLI_CONTROLLER
        self._subscribers: list[asyncio.Queue[SessionEvent]] = []
        self._state_condition = asyncio.Condition()

    @property
    def state(self) -> SessionState:
        return SessionState(
            promptable=self._run_task is None,
            controller=self._active_controller,
            running=self._run_task is not None,
        )

    @property
    def history(self) -> list[BaseMessage]:
        return list(self._history)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator[asyncio.Queue[SessionEvent]]:
        queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(StateChangedEvent(state=self.state))
        try:
            yield queue
        finally:
            with suppress(ValueError):
                self._subscribers.remove(queue)

    async def wait_for_controller_change(self, *, controller: SessionControllerName) -> SessionControllerName:
        async with self._state_condition:
            await self._state_condition.wait_for(lambda: self._active_controller != controller)
            return self._active_controller

    async def activate_controller(self, controller: SessionControllerName) -> bool:
        if self._active_controller == controller:
            return False

        self._active_controller = controller
        await self._publish_state()
        return True

    async def restore_cli_controller(self) -> None:
        if self._active_controller != CLI_CONTROLLER and self._run_task is not None:
            await self.cancel_current_run()

        if self._active_controller == CLI_CONTROLLER:
            return

        self._active_controller = CLI_CONTROLLER
        await self._publish_state()

    async def submit_prompt(
        self,
        *,
        controller: SessionControllerName,
        content: str | list[dict[str, Any]],
    ) -> PromptSubmissionResult:
        if controller != self._active_controller:
            return PromptSubmissionResult(accepted=False, reason="inactive_controller")
        return await self._submit_prompt(content)

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

    async def _submit_prompt(self, content: str | list[dict[str, Any]]) -> PromptSubmissionResult:
        if self._run_task is not None:
            return PromptSubmissionResult(accepted=False, reason="not_ready")

        self._history.append(UserMessage(content=content))
        self._run_task = asyncio.create_task(self._run_until_boundary())
        await self._publish_state()
        return PromptSubmissionResult(accepted=True, reason="accepted")

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
        async with self._state_condition:
            self._state_condition.notify_all()

    def _publish_event(self, event: SessionEvent) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(event)


def _get_latest_assistant_summary(history: Sequence[BaseMessage]) -> str:
    for message in reversed(history):
        if isinstance(message, AssistantMessage) and message.content:
            return message.content
    return ""
