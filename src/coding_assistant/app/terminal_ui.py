from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable
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

from coding_assistant.app.output import (
    DeltaRenderer,
    format_prompt_preview,
    format_session_status,
    print_active_prompt,
    print_session_status,
    print_system_message,
    print_tool_calls,
)
from coding_assistant.core.agent_session import (
    AgentSession,
    PromptAcceptedEvent,
    PromptStartedEvent,
    RunCancelledEvent,
    RunFailedEvent,
    RunFinishedEvent,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.llm.types import (
    CompletionEvent,
    ContentDeltaEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    SystemMessage,
)


PromptSubmitHandler = Callable[[str], Awaitable[bool]]


class SlashCompleter(Completer):
    """Autocomplete only slash-prefixed commands."""

    def __init__(self, words: list[str]):
        self.pattern = re.compile(r"/[a-zA-Z0-9_]*")
        self.word_completer = WordCompleter(words, pattern=self.pattern)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        """Yield slash-command completions when the prompt starts with `/`."""
        if document.text_before_cursor.startswith("/"):
            yield from self.word_completer.get_completions(document, complete_event)


def create_prompt_history(history_path: Path) -> FileHistory:
    """Create the file-backed history used by the interactive terminal UI."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    return FileHistory(str(history_path))


def format_queued_prompts(session: AgentSession) -> str:
    """Return the queued-prompt widget text shown above the input line."""
    state = session.state
    if not state.pending_prompts:
        return ""

    lines: list[str] = []
    for prompt in state.pending_prompts[:2]:
        lines.append(f"↳ {format_prompt_preview(prompt)}")

    remaining_count = state.queued_prompt_count - min(len(state.pending_prompts), 2)
    if remaining_count > 0:
        lines.append(f"↳ +{remaining_count} more")

    return "\n".join(lines)


def create_terminal_application(
    *,
    session: AgentSession,
    history_path: Path,
    words: list[str] | None = None,
) -> tuple[Application[None], asyncio.Queue[str]]:
    """Build the shared non-fullscreen terminal UI and its submission queue."""
    queued_window = ConditionalContainer(
        content=Window(
            content=FormattedTextControl(
                text=lambda: [("class:queued", format_queued_prompts(session))],
                show_cursor=False,
            ),
            dont_extend_height=True,
        ),
        filter=Condition(lambda: bool(session.state.pending_prompts)),
    )
    footer_window = Window(
        content=FormattedTextControl(
            text=lambda: [("class:footer", format_session_status(session.state))],
            show_cursor=False,
        ),
        height=Dimension.exact(1),
    )

    key_bindings = KeyBindings()
    answer_queue: asyncio.Queue[str] = asyncio.Queue()

    @key_bindings.add("c-c")
    def exit_on_interrupt(event: Any) -> None:
        event.app.exit()

    completer = SlashCompleter(words) if words else None

    def accept_buffer(buffer: Buffer) -> bool:
        answer_queue.put_nowait(buffer.text)
        return False

    input_buffer = Buffer(
        completer=completer,
        complete_while_typing=True,
        history=create_prompt_history(history_path),
        multiline=False,
        accept_handler=accept_buffer,
    )
    input_window = Window(
        content=BufferControl(buffer=input_buffer),
        height=Dimension.exact(1),
    )
    input_row = VSplit(
        [
            Window(
                content=FormattedTextControl([("class:prompt", "> ")], show_cursor=False),
                width=Dimension.exact(2),
                dont_extend_height=True,
            ),
            input_window,
        ],
        padding=0,
    )

    @key_bindings.add("c-d")
    def exit_on_eof(event: Any) -> None:
        if not input_buffer.text:
            event.app.exit()

    layout = Layout(HSplit([queued_window, input_row, footer_window], padding=0), focused_element=input_window)

    application: Application[None] = Application(
        layout=layout,
        key_bindings=key_bindings,
        style=Style.from_dict(
            {
                "prompt": "#bbbbbb",
                "queued": "#888888",
                "footer": "#888888",
            }
        ),
        full_screen=False,
        refresh_interval=0.1,
    )
    return application, answer_queue


async def run_session_output(
    *,
    session: AgentSession,
    system_message: SystemMessage,
    show_state_updates: bool = False,
    show_prompt_accepted: bool = False,
) -> None:
    """Render one session's streamed events to the local terminal."""
    renderer = DeltaRenderer()
    last_state_summary: str | None = None
    print_system_message(system_message)

    async with session.subscribe() as queue:
        try:
            while True:
                event = await queue.get()
                if isinstance(event, ContentDeltaEvent):
                    renderer.on_delta(event.content)
                    continue
                if isinstance(event, PromptAcceptedEvent):
                    if not show_prompt_accepted:
                        continue
                    renderer.finish()
                    print_active_prompt(event.content)
                    continue
                if isinstance(event, PromptStartedEvent):
                    renderer.finish()
                    print_active_prompt(event.content)
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
                    rich_print(f"[bold red]Run failed:[/bold red] {event.error}")
                    continue
                if isinstance(event, StateChangedEvent):
                    if not show_state_updates:
                        continue
                    state_summary = format_session_status(event.state)
                    if state_summary == last_state_summary:
                        continue
                    last_state_summary = state_summary
                    renderer.finish(trailing_blank_line=False)
                    print_session_status(event.state)
                    continue
                if isinstance(event, (ReasoningDeltaEvent, StatusEvent, CompletionEvent)):
                    continue
        finally:
            renderer.finish()


async def run_terminal_ui(
    *,
    session: AgentSession,
    system_message: SystemMessage,
    history_path: Path,
    words: list[str] | None = None,
    submit_handler: PromptSubmitHandler,
) -> None:
    """Run the shared interactive terminal UI."""
    application, answer_queue = create_terminal_application(
        session=session,
        history_path=history_path,
        words=words,
    )
    transcript_task = asyncio.create_task(
        run_session_output(
            session=session,
            system_message=system_message,
        )
    )

    with patch_stdout(raw=True):
        application_task = asyncio.create_task(application.run_async())
        next_answer_task: asyncio.Task[str] = asyncio.create_task(answer_queue.get())

        try:
            while True:
                done, _ = await asyncio.wait(
                    {application_task, next_answer_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if application_task in done:
                    await application_task
                    return

                answer = next_answer_task.result()
                should_exit = await submit_handler(answer)
                application.invalidate()
                if should_exit:
                    if not application.is_done:
                        application.exit()
                    await application_task
                    return

                next_answer_task = asyncio.create_task(answer_queue.get())
        finally:
            next_answer_task.cancel()
            if not application_task.done():
                application.exit()
            await asyncio.gather(application_task, return_exceptions=True)
            transcript_task.cancel()
            await asyncio.gather(transcript_task, return_exceptions=True)
