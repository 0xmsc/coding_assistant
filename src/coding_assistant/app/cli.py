from __future__ import annotations

import asyncio
import re
from argparse import Namespace
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich import print as rich_print

from coding_assistant.app.default_agent import (
    build_default_agent_config,
    build_initial_system_message,
    create_default_agent,
)
from coding_assistant.app.image import get_image
from coding_assistant.app.output import (
    DeltaRenderer,
    format_prompt_preview,
    format_session_status,
    print_active_prompt,
    print_info_message,
    print_session_status,
    print_system_message,
    print_tool_calls,
)
from coding_assistant.core.agent_session import (
    AgentSession,
    PromptStartedEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.infra.paths import get_app_cache_dir
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.llm.types import (
    ContentDeltaEvent,
    StatusEvent,
    SystemMessage,
)
from coding_assistant.remote.registry import register_remote_instance
from coding_assistant.remote.server import start_worker_server


class PromptSubmitType(Enum):
    """How a prompt was submitted."""

    STEERING = auto()  # Enter - injected into active run at next boundary
    QUEUED = auto()  # TAB - added to back of queue


CLI_COMMANDS = ["/exit", "/help", "/compact", "/image"]


class SlashCompleter(Completer):
    """Autocomplete slash-prefixed commands."""

    def __init__(self, words: list[str]):
        self.pattern = re.compile(r"/[a-zA-Z0-9_]*")
        self.word_completer = WordCompleter(words, pattern=self.pattern)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        if document.text_before_cursor.startswith("/"):
            yield from self.word_completer.get_completions(document, complete_event)


def _create_history(history_path: Path) -> FileHistory:
    """Create file-backed prompt history."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(history_path))


def _format_queued_prompts(session: AgentSession) -> str:
    """Format pending prompts for display above input line."""
    state = session.state
    if not state.pending_prompts:
        return ""

    lines: list[str] = []
    for prompt in state.pending_prompts[:2]:
        lines.append(f"↳ {format_prompt_preview(prompt)}")

    remaining = state.queued_prompt_count - min(len(state.pending_prompts), 2)
    if remaining > 0:
        lines.append(f"↳ +{remaining} more")

    return "\n".join(lines)


async def run_cli(args: Namespace) -> None:
    """Run the interactive CLI."""
    config = build_default_agent_config(args)

    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = build_initial_system_message(instructions=bundle.instructions)
        session = AgentSession(
            history=[system_message],
            model=args.model,
            tools=bundle.tools,
        )
        try:
            async with start_worker_server(session=session) as worker_server:
                async with register_remote_instance(endpoint=worker_server.endpoint):
                    rich_print(f"[dim]Remote endpoint:[/dim] {worker_server.endpoint}")
                    await _run_ui(
                        session=session,
                        system_message=system_message,
                        history_path=get_app_cache_dir() / "history",
                    )
        finally:
            await session.close()


async def _run_ui(
    *,
    session: AgentSession,
    system_message: SystemMessage,
    history_path: Path,
) -> None:
    """Run the terminal UI loop."""
    application, answer_queue = _create_application(session, history_path)

    output_task = asyncio.create_task(_run_output(session=session, system_message=system_message))

    with patch_stdout(raw=True):
        app_task = asyncio.create_task(application.run_async())
        get_next = asyncio.create_task(answer_queue.get())

        try:
            while True:
                done, _ = await asyncio.wait({app_task, get_next}, return_when=asyncio.FIRST_COMPLETED)

                if app_task in done:
                    return

                submission = await get_next
                if await _handle_submission(session=session, content=submission.content, submit_type=submission.type):
                    application.exit()
                    await app_task
                    return

                get_next = asyncio.create_task(answer_queue.get())
        finally:
            get_next.cancel()
            if not app_task.done():
                application.exit()
            await asyncio.gather(app_task, return_exceptions=True)
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)


def _create_application(
    session: AgentSession, history_path: Path
) -> tuple[Application[None], asyncio.Queue[PromptSubmission]]:
    """Build the terminal UI."""
    queued_window = _build_queued_window(session)
    footer = _build_footer(session)

    key_bindings = KeyBindings()
    answer_queue: asyncio.Queue[PromptSubmission] = asyncio.Queue()
    submit_type = [PromptSubmitType.QUEUED]
    input_buffer: list[Buffer] = []

    @key_bindings.add("c-c")
    async def handle_interrupt(event: Any) -> None:
        state = session.state
        if state.running:
            await session.cancel_current_run(pause_queue=True)
        elif state.paused:
            await session.resume()
        elif input_buffer[0].text:
            input_buffer[0].text = ""
            input_buffer[0].cursor_position = 0
        event.app.invalidate()

    @key_bindings.add("tab")
    def submit_tab(_: Any) -> None:
        submit_type[0] = PromptSubmitType.QUEUED
        input_buffer[0].validate_and_handle()

    @key_bindings.add("c-m")
    def submit_enter(_: Any) -> None:
        submit_type[0] = PromptSubmitType.STEERING
        input_buffer[0].validate_and_handle()

    @key_bindings.add("c-j")
    def insert_newline(_: Any) -> None:
        input_buffer[0].insert_text("\n")

    @key_bindings.add("c-u")
    def unqueue_last(_: Any) -> None:
        asyncio.get_running_loop().create_task(_unqueue_to_buffer(session, input_buffer[0]))

    @key_bindings.add("c-d")
    def exit_on_eof(event: Any) -> None:
        if not input_buffer[0].text:
            event.app.exit()

    def accept(buff: Buffer) -> bool:
        st = submit_type[0]
        submit_type[0] = PromptSubmitType.QUEUED
        answer_queue.put_nowait(PromptSubmission(content=buff.text, type=st))
        return False

    buff = Buffer(
        completer=SlashCompleter(CLI_COMMANDS),
        complete_while_typing=True,
        history=_create_history(history_path),
        multiline=True,
        accept_handler=accept,
    )
    input_buffer.append(buff)

    layout = Layout(
        HSplit(
            [
                queued_window,
                VSplit(
                    [
                        Window(
                            FormattedTextControl([("class:prompt", "> ")], show_cursor=False), width=Dimension.exact(2)
                        ),
                        Window(BufferControl(buffer=buff), height=Dimension(min=1, max=10), wrap_lines=True),
                    ],
                ),
                footer,
            ],
        ),
        focused_element=buff,
    )

    return (
        Application(
            layout=layout,
            key_bindings=key_bindings,
            style=Style.from_dict({"prompt": "#bbbbbb", "queued": "#888888", "footer": "#888888"}),
            full_screen=False,
            refresh_interval=0.1,
        ),
        answer_queue,
    )


def _build_queued_window(session: AgentSession) -> ConditionalContainer:
    return ConditionalContainer(
        content=Window(
            content=FormattedTextControl(
                text=lambda: [("class:queued", _format_queued_prompts(session))],
                show_cursor=False,
            ),
            dont_extend_height=True,
        ),
        filter=Condition(lambda: bool(session.state.pending_prompts)),
    )


def _build_footer(session: AgentSession) -> Window:
    return Window(
        content=FormattedTextControl(
            text=lambda: [("class:footer", format_session_status(session.state))],
            show_cursor=False,
        ),
        height=Dimension.exact(1),
    )


async def _unqueue_to_buffer(session: AgentSession, buf: Buffer) -> None:
    """Move last queued prompt back to input buffer."""
    content = await session.pop_last_queued_prompt()
    if content is not None:
        buf.text = f"{content}\n{buf.text}" if buf.text else str(content)
        buf.cursor_position = len(buf.text)


async def _run_output(
    *,
    session: AgentSession,
    system_message: SystemMessage,
    show_state_updates: bool = False,
) -> None:
    """Display session output to terminal."""
    renderer = DeltaRenderer()
    last_summary: str | None = None
    print_system_message(system_message)

    async with session.subscribe() as queue:
        try:
            while True:
                event = await queue.get()
                if isinstance(event, ContentDeltaEvent):
                    renderer.on_delta(event.content)
                elif isinstance(event, PromptStartedEvent):
                    renderer.finish()
                    print_active_prompt(event.content)
                elif isinstance(event, ToolCallsEvent):
                    renderer.finish(trailing_blank_line=False)
                    rich_print()
                    print_tool_calls(event.message)
                elif isinstance(event, RunFinishedEvent):
                    renderer.finish()
                elif isinstance(event, RunCancelledEvent):
                    renderer.finish()
                elif isinstance(event, RunFailedEvent):
                    renderer.finish()
                    rich_print(f"[bold red]Run failed:[/bold red] {event.error}")
                elif isinstance(event, StateChangedEvent):
                    if not show_state_updates:
                        continue
                    summary = format_session_status(event.state)
                    if summary != last_summary:
                        last_summary = summary
                        renderer.finish(trailing_blank_line=False)
                        print_session_status(event.state)
                elif isinstance(event, StatusEvent):
                    renderer.finish(trailing_blank_line=False)
                    print_info_message(event.message)
        finally:
            renderer.finish()


@dataclass
class PromptSubmission:
    """A submitted prompt with its submission type."""

    content: str
    type: PromptSubmitType


async def _handle_submission(*, session: AgentSession, content: str, submit_type: PromptSubmitType) -> bool:
    """Handle one prompt and return True to exit."""
    stripped = content.strip()
    if stripped == "/exit":
        return True
    if stripped == "/help":
        rich_print("Available commands:\n  /exit\n  /help\n  /compact\n  /image <path-or-url>")
        return False
    if stripped == "/compact":
        return not await session.enqueue_prompt(
            "Immediately compact our conversation so far by using the `compact_conversation` tool.",
        )
    if stripped.startswith("/image"):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            rich_print("[red]/image requires a path or URL[/red]")
            return False
        data_url = await get_image(parts[1])
        image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
        return not await session.enqueue_prompt(image_content)

    if submit_type == PromptSubmitType.STEERING:
        return not await session.enqueue_steering_prompt(content)
    else:
        return not await session.enqueue_prompt(content)
