from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from rich import print

from coding_assistant.core.agent import run_agent_event_stream
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.tool_calls import execute_tool_calls
from coding_assistant.app.cli import (
    CLI_COMMAND_NAMES,
    _build_initial_system_message,
    _create_prompt_session,
    _prompt_with_session,
    build_default_agent_config,
    create_default_agent,
    handle_cli_input,
)
from coding_assistant.app.output import DeltaRenderer, print_system_message, print_tool_calls
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.llm.openai import stream_completion as openai_stream_completion
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    CompletionEvent,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    SystemMessage,
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
    def state(self) -> SessionState:
        """Return the current promptability, controller ownership, and run status."""

        ...

    @property
    def history(self) -> list[BaseMessage]:
        """Return a snapshot of the current conversation history."""

        ...

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[SessionEvent]]:
        """Subscribe to session events; new subscribers receive the current state first."""

        ...

    async def wait_for_controller_change(self, *, controller: SessionControllerName) -> SessionControllerName:
        """Wait until the active controller differs from `controller` and return the new owner."""

        ...

    async def activate_controller(self, controller: SessionControllerName) -> bool:
        """Switch input ownership to `controller`; return `False` if it already owns input."""

        ...

    async def restore_cli_controller(self) -> None:
        """Return input ownership to the CLI, cancelling an in-flight remote run if needed."""

        ...

    async def submit_prompt(
        self,
        *,
        controller: SessionControllerName,
        content: str | list[dict[str, Any]],
    ) -> PromptSubmissionResult:
        """Submit a prompt if `controller` still owns input, reporting why rejection happened."""

        ...

    async def cancel_current_run(self) -> bool:
        """Cancel the active run and return whether any run was actually cancelled."""

        ...


class SessionApp:
    """Own the interactive session lifecycle and switch input control between CLI and remote."""

    def __init__(
        self,
        *,
        system_message: SystemMessage,
        history: Sequence[BaseMessage],
        model: str,
        tools: Sequence[Tool],
        prompt_user: Callable[[list[str] | None], Coroutine[object, object, str]],
        working_directory: Path,
        set_local_worker_endpoint: Callable[[str], None],
        close_tools: Callable[[], Awaitable[None]],
        completion_streamer: CompletionStreamer | None = None,
    ) -> None:
        self._system_message = system_message
        self._history = list(history)
        self._model = model
        self._tools = list(tools)
        self._prompt_user = prompt_user
        self._working_directory = working_directory
        self._set_local_worker_endpoint = set_local_worker_endpoint
        self._close_tools = close_tools
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

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[SessionEvent]]:
        return self._subscribe()

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

    async def run(self) -> None:
        from coding_assistant.remote.server import start_worker_server

        print_system_message(self._system_message)

        async with start_worker_server(session=self, cwd=self._working_directory) as worker_server:
            self._set_local_worker_endpoint(worker_server.endpoint)
            try:
                await self._drive_cli()
            finally:
                await self._close_tools()

    async def _drive_cli(self) -> None:
        renderer = DeltaRenderer()

        async with self.subscribe() as queue:
            while True:
                state = self.state
                if state.promptable and state.controller == CLI_CONTROLLER:
                    answer = await _prompt_while_controller_is_active(
                        session=self,
                        controller=CLI_CONTROLLER,
                        prompt_user=self._prompt_user,
                        words=CLI_COMMAND_NAMES,
                    )
                    if answer is None:
                        continue
                    if await handle_cli_input(
                        answer=answer,
                        renderer=renderer,
                        submit_prompt_or_warn=lambda content: _submit_prompt_or_warn(
                            session=self,
                            controller=CLI_CONTROLLER,
                            content=content,
                        ),
                    ):
                        return
                    continue

                event = await queue.get()
                if isinstance(event, ContentDeltaEvent):
                    renderer.on_delta(event.content)
                    continue
                if isinstance(event, ToolCallsEvent):
                    renderer.finish(trailing_blank_line=False)
                    print_tool_calls(event.message)
                    continue
                if isinstance(event, RunFinishedEvent):
                    renderer.finish()
                    continue
                if isinstance(event, RunCancelledEvent):
                    renderer.finish()
                    continue
                if isinstance(event, RunFailedEvent):
                    renderer.finish()
                    print(f"[bold red]Run failed:[/bold red] {event.error}")
                    continue
                if isinstance(event, (StateChangedEvent, ReasoningDeltaEvent, StatusEvent, CompletionEvent)):
                    continue

    @asynccontextmanager
    async def _subscribe(self) -> AsyncIterator[asyncio.Queue[SessionEvent]]:
        queue: asyncio.Queue[SessionEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        queue.put_nowait(StateChangedEvent(state=self.state))
        try:
            yield queue
        finally:
            with suppress(ValueError):
                self._subscribers.remove(queue)

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


async def _submit_prompt_or_warn(
    *,
    session: SessionControlSurface,
    controller: SessionControllerName,
    content: str | list[dict[str, Any]],
) -> bool:
    result = await session.submit_prompt(controller=controller, content=content)
    if result.accepted:
        return True
    if result.reason == "inactive_controller":
        print("Remote control took ownership before the local prompt was submitted.")
        return False
    print("The session is not ready for another local prompt yet.")
    return False


async def _prompt_while_controller_is_active(
    *,
    session: SessionControlSurface,
    controller: SessionControllerName,
    prompt_user: Callable[[list[str] | None], Coroutine[object, object, str]],
    words: list[str] | None,
) -> str | None:
    prompt_task = asyncio.create_task(prompt_user(words))
    controller_task = asyncio.create_task(session.wait_for_controller_change(controller=controller))
    done, pending = await asyncio.wait(
        {prompt_task, controller_task},
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    if controller_task in done:
        prompt_task.cancel()
        try:
            await prompt_task
        except asyncio.CancelledError:
            pass
        return None

    controller_task.cancel()
    try:
        await controller_task
    except asyncio.CancelledError:
        pass
    return await prompt_task


async def run_session_app(args: Namespace) -> None:
    config = build_default_agent_config(args)
    prompt_session = _create_prompt_session()

    async def prompt_user(words: list[str] | None = None) -> str:
        return await _prompt_with_session(prompt_session, words=words)

    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = _build_initial_system_message(instructions=bundle.instructions)
        app = SessionApp(
            system_message=system_message,
            history=[system_message],
            model=args.model,
            tools=bundle.tools,
            prompt_user=prompt_user,
            working_directory=config.working_directory,
            set_local_worker_endpoint=bundle.set_local_worker_endpoint,
            close_tools=bundle.close_tools,
        )
        await app.run()


def _get_latest_assistant_summary(history: Sequence[BaseMessage]) -> str:
    for message in reversed(history):
        if isinstance(message, AssistantMessage) and message.content:
            return message.content
    return ""
