from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from rich import print as rich_print

from coding_assistant.cli.agent import build_cli_agent_config, create_cli_agent
from coding_assistant.cli.commands import (
    SlashCommand,
    execute_slash_command,
    get_default_commands,
)
from coding_assistant.cli.output import (
    StreamRenderer,
    format_prompt_preview,
    format_session_status,
    print_active_prompt,
    print_info_message,
    print_system_message,
    print_tool_calls,
    ring_bell,
)
from coding_assistant.core.runtime import build_initial_system_message
from coding_assistant.core.agent_session import (
    AgentSession,
    AgentSessionEvent,
    AgentSessionProtocol,
    AssistantMessageCompletedEvent,
    AssistantMessageDeltaEvent,
    HistoryResetEvent,
    PromptStartedEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
)
from coding_assistant.core.compacting_session import AutoCompactingSession
from coding_assistant.infra.paths import get_app_cache_dir
from coding_assistant.llm.context_window import resolve_auto_compaction_budget
from coding_assistant.llm.types import (
    ModelRetryEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    SystemMessage,
    UserMessage,
)


class PromptSubmitType(Enum):
    """How a prompt was submitted."""

    STEERING = auto()  # Enter - injected into active run at next boundary
    QUEUED = auto()  # TAB - added to back of queue


CLI_COMMANDS = [cmd.name for cmd in get_default_commands()]


class SlashCompleter(Completer):
    """Autocomplete slash-prefixed commands."""

    def __init__(self, commands: Sequence[SlashCommand | str]):
        self._commands = commands

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if text.startswith("/"):
            query = text.lower()
            for cmd in self._commands:
                name = cmd.name if isinstance(cmd, SlashCommand) else cmd
                desc = cmd.description if isinstance(cmd, SlashCommand) else None
                if name.lower().startswith(query):
                    yield Completion(
                        text=name,
                        start_position=-len(text),
                        display=name,
                        display_meta=desc,
                    )


def _create_history(history_path: Path) -> FileHistory:
    """Create file-backed prompt history."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(history_path))


def _format_queued_prompts(session: AgentSessionProtocol) -> str:
    """Format pending prompts for display above input line."""
    state = session.state
    if not state.pending_prompts:
        return ""

    lines: list[str] = []
    for prompt in state.pending_prompts[:2]:
        lines.append(f"↳ {format_prompt_preview(prompt)}")

    remaining = max(0, len(state.pending_prompts) - 2)
    if remaining > 0:
        lines.append(f"↳ +{remaining} more")

    return "\n".join(lines)


async def run_cli(args: Namespace) -> None:
    """Run the interactive CLI."""
    config = build_cli_agent_config(args)

    async with create_cli_agent(config=config) as bundle:
        system_message = build_initial_system_message(instructions=bundle.instructions)
        raw_session = AgentSession(
            history=[system_message],
            model=args.model,
            tools=bundle.tools,
        )
        session: AgentSessionProtocol
        if getattr(args, "auto_compact", True):
            token_budget = resolve_auto_compaction_budget(
                args.model,
                configured_budget=getattr(args, "auto_compact_token_budget", None),
            )
            session = AutoCompactingSession(
                raw_session,
                token_budget=token_budget,
            )
        else:
            session = raw_session
        try:
            await _run_ui(
                session=session,
                system_message=system_message,
                history_path=get_app_cache_dir() / "history",
                bell=getattr(args, "bell", True),
            )
        finally:
            await session.close()


async def _run_ui(
    *,
    session: AgentSessionProtocol,
    system_message: SystemMessage,
    history_path: Path,
    bell: bool = True,
) -> None:
    """Run the terminal UI loop."""
    application, answer_queue = _create_application(session, history_path)

    output_task = asyncio.create_task(_run_output(session=session, system_message=system_message, bell=bell))

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
    session: AgentSessionProtocol, history_path: Path
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

    commands = get_default_commands()
    buff = Buffer(
        completer=SlashCompleter(commands),
        complete_while_typing=True,
        history=_create_history(history_path),
        multiline=True,
        accept_handler=accept,
    )
    input_buffer.append(buff)

    layout = Layout(
        FloatContainer(
            content=HSplit(
                [
                    Window(height=Dimension.exact(1)),
                    queued_window,
                    VSplit(
                        [
                            Window(
                                FormattedTextControl([("class:prompt", "> ")], show_cursor=False),
                                width=Dimension.exact(2),
                            ),
                            Window(BufferControl(buffer=buff), height=Dimension(min=1, max=10), wrap_lines=True),
                        ],
                    ),
                    footer,
                ],
            ),
            floats=[
                Float(
                    xcursor=True,
                    ycursor=True,
                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                ),
            ],
        ),
        focused_element=buff,
    )

    return (
        Application(
            layout=layout,
            key_bindings=key_bindings,
            style=Style.from_dict(
                {
                    "prompt": "#bbbbbb",
                    "queued": "#888888",
                    "footer": "#888888",
                    "completion-menu": "bg:#2b2b2b #ffffff",
                    "completion-menu.completion": "bg:#2b2b2b #cccccc",
                    "completion-menu.completion.current": "bg:#444444 #ffffff bold",
                    "completion-menu.meta": "bg:#2b2b2b #888888",
                    "completion-menu.meta.completion.current": "bg:#444444 #aaaaaa",
                }
            ),
            full_screen=False,
            refresh_interval=0.1,
        ),
        answer_queue,
    )


def _build_queued_window(session: AgentSessionProtocol) -> ConditionalContainer:
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


def _build_footer(session: AgentSessionProtocol) -> Window:
    model = getattr(session, "model", None)
    model_str = model if isinstance(model, str) else None
    return Window(
        content=FormattedTextControl(
            text=lambda: [("class:footer", format_session_status(session.state, model=model_str))],
            show_cursor=False,
        ),
        height=Dimension.exact(1),
    )


async def _unqueue_to_buffer(session: AgentSessionProtocol, buf: Buffer) -> None:
    """Move last queued prompt back to input buffer."""
    content = await session.pop_last_queued_prompt()
    if content is not None:
        buf.text = f"{content}\n{buf.text}" if buf.text else str(content)
        buf.cursor_position = len(buf.text)


async def _run_output(
    *,
    session: AgentSessionProtocol,
    system_message: SystemMessage,
    bell: bool = True,
) -> None:
    """Display session output to terminal."""
    renderer = StreamRenderer()
    printed_tool_call_messages: set[int] = set()
    print_system_message(system_message)

    async with session.subscribe() as queue:
        try:
            while True:
                event = await queue.get()
                if isinstance(event, ModelRetryEvent):
                    renderer.retry()
                    continue
                _render_agent_event(
                    event=event,
                    renderer=renderer,
                    printed_tool_call_messages=printed_tool_call_messages,
                    bell=bell,
                )
        finally:
            renderer.finish()


def _render_agent_event(
    *,
    event: AgentSessionEvent,
    renderer: StreamRenderer,
    printed_tool_call_messages: set[int],
    bell: bool = True,
) -> None:
    if isinstance(event, AssistantMessageDeltaEvent):
        renderer.on_content_delta(event.content)
        return
    if isinstance(event, ReasoningDeltaEvent):
        renderer.on_reasoning_delta(event.content)
        return
    if isinstance(event, PromptStartedEvent):
        renderer.finish()
        print_active_prompt(event.content)
        return
    if isinstance(event, HistoryResetEvent):
        renderer.finish()
        for message in event.history:
            if isinstance(message, UserMessage):
                print_active_prompt(message.content)
        return
    if isinstance(event, AssistantMessageCompletedEvent) and event.completion.message.tool_calls:
        message = event.completion.message
        message_id = id(message)
        if message_id in printed_tool_call_messages:
            return
        printed_tool_call_messages.add(message_id)
        renderer.finish()
        print_tool_calls(message)
        return
    if isinstance(event, RunFinishedEvent):
        renderer.finish()
        if bell:
            ring_bell()
        return
    if isinstance(event, RunCancelledEvent):
        renderer.finish()
        return
    if isinstance(event, RunFailedEvent):
        renderer.finish()
        rich_print(f"[bold red]Run failed:[/bold red] {event.error}")
        return
    if isinstance(event, StatusEvent):
        print_info_message(event.message)


@dataclass
class PromptSubmission:
    """A submitted prompt with its submission type."""

    content: str
    type: PromptSubmitType


async def _handle_submission(
    *,
    session: AgentSessionProtocol,
    content: str,
    submit_type: PromptSubmitType,
    commands: Sequence[SlashCommand] | None = None,
) -> bool:
    """Handle one prompt and return True to exit."""
    active_commands = commands if commands is not None else get_default_commands()
    command_result = await execute_slash_command(
        commands=active_commands,
        content=content,
        session=session,
    )
    if command_result is not None:
        return command_result

    if submit_type == PromptSubmitType.STEERING:
        return not await session.enqueue_steering_prompt(content)
    else:
        return not await session.enqueue_prompt(content)
