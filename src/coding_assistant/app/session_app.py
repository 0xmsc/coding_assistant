from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import AbstractAsyncContextManager, suppress
from pathlib import Path
from typing import Any

from rich import print

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
from coding_assistant.app.session_host import (
    CLI_CONTROLLER,
    PromptSubmissionResult,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    SessionControlSurface,
    SessionControllerName,
    SessionEvent,
    SessionHost,
    SessionState,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.llm.types import (
    BaseMessage,
    CompletionEvent,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    SystemMessage,
)
from coding_assistant.remote.server import start_worker_server


class SessionApp:
    """Own the interactive session lifecycle and switch input control between CLI and remote."""

    def __init__(
        self,
        *,
        system_message: SystemMessage,
        session: SessionHost,
        prompt_user: Callable[[list[str] | None], Coroutine[object, object, str]],
        working_directory: Path,
        set_local_worker_endpoint: Callable[[str], None],
        close_tools: Callable[[], Awaitable[None]],
    ) -> None:
        self._system_message = system_message
        self._session = session
        self._prompt_user = prompt_user
        self._working_directory = working_directory
        self._set_local_worker_endpoint = set_local_worker_endpoint
        self._close_tools = close_tools

    @property
    def state(self) -> SessionState:
        return self._session.state

    @property
    def history(self) -> list[BaseMessage]:
        return self._session.history

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[SessionEvent]]:
        return self._session.subscribe()

    async def wait_for_controller_change(self, *, controller: SessionControllerName) -> SessionControllerName:
        return await self._session.wait_for_controller_change(controller=controller)

    async def activate_controller(self, controller: SessionControllerName) -> bool:
        return await self._session.activate_controller(controller)

    async def restore_cli_controller(self) -> None:
        await self._session.restore_cli_controller()

    async def submit_prompt(
        self,
        *,
        controller: SessionControllerName,
        content: str | list[dict[str, Any]],
    ) -> PromptSubmissionResult:
        return await self._session.submit_prompt(controller=controller, content=content)

    async def cancel_current_run(self) -> bool:
        return await self._session.cancel_current_run()

    async def run(self) -> None:
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
        session = SessionHost(
            history=[system_message],
            model=args.model,
            tools=bundle.tools,
        )
        app = SessionApp(
            system_message=system_message,
            session=session,
            prompt_user=prompt_user,
            working_directory=config.working_directory,
            set_local_worker_endpoint=bundle.set_local_worker_endpoint,
            close_tools=bundle.close_tools,
        )
        await app.run()
