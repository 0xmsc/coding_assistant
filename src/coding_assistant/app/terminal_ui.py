from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable
from enum import Enum
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
from prompt_toolkit.layout.containers import AnyContainer, ConditionalContainer
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from coding_assistant.app.output import format_prompt_preview, format_session_status, run_session_output
from coding_assistant.core.agent_session import AgentSession
from coding_assistant.llm.types import SystemMessage


class TerminalUiMode(str, Enum):
    INTERACTIVE = "interactive"
    READONLY = "readonly"


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
    mode: TerminalUiMode,
    history_path: Path | None,
    words: list[str] | None = None,
) -> tuple[Application[None], asyncio.Queue[str] | None]:
    """Build the shared non-fullscreen terminal UI and its submission queue when interactive."""
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
    input_row: AnyContainer | None = None
    input_window: Window | None = None
    answer_queue: asyncio.Queue[str] | None = None

    @key_bindings.add("c-c")
    def exit_on_interrupt(event: Any) -> None:
        event.app.exit()

    if mode is TerminalUiMode.INTERACTIVE:
        if history_path is None:
            raise ValueError("Interactive terminal UI requires a history path.")

        completer = SlashCompleter(words) if words else None
        answer_queue = asyncio.Queue()

        def accept_buffer(buffer: Buffer) -> bool:
            assert answer_queue is not None
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

    body: list[AnyContainer] = [queued_window]
    if input_row is not None:
        body.append(input_row)
    body.append(footer_window)

    layout = (
        Layout(HSplit(body, padding=0), focused_element=input_window)
        if input_window
        else Layout(HSplit(body, padding=0))
    )

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


async def run_terminal_ui(
    *,
    session: AgentSession,
    system_message: SystemMessage,
    mode: TerminalUiMode,
    history_path: Path | None = None,
    words: list[str] | None = None,
    submit_handler: PromptSubmitHandler | None = None,
) -> None:
    """Run the shared terminal UI for either interactive CLI or readonly worker mode."""
    if mode is TerminalUiMode.INTERACTIVE and submit_handler is None:
        raise ValueError("Interactive terminal UI requires a submit handler.")

    application, answer_queue = create_terminal_application(
        session=session,
        mode=mode,
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
        next_answer_task: asyncio.Task[str] | None = None
        if answer_queue is not None:
            next_answer_task = asyncio.create_task(answer_queue.get())

        try:
            if mode is TerminalUiMode.READONLY:
                await application_task
                return

            assert submit_handler is not None
            assert answer_queue is not None
            assert next_answer_task is not None
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
            if next_answer_task is not None:
                next_answer_task.cancel()
            if not application_task.done():
                application.exit()
            await asyncio.gather(application_task, return_exceptions=True)
            transcript_task.cancel()
            await asyncio.gather(transcript_task, return_exceptions=True)
