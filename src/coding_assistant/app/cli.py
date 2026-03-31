from __future__ import annotations

import asyncio
import re
from argparse import Namespace
from collections.abc import Iterable

from coding_assistant.app.default_agent import (
    build_default_agent_config,
    build_initial_system_message,
    create_default_agent,
)
from coding_assistant.app.image import get_image
from coding_assistant.app.output import format_session_status, run_session_output
from coding_assistant.core.agent_session import AgentSession
from coding_assistant.infra.paths import get_app_cache_dir
from coding_assistant.integrations.mcp_client import print_mcp_tools
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich import print
from rich.console import Console
from rich.rule import Rule

CLI_COMMAND_NAMES = ["/exit", "/help", "/compact", "/image", "/priority", "/interrupt"]


class SlashCompleter(Completer):
    """Autocomplete only slash-prefixed commands."""

    def __init__(self, words: list[str]):
        self.pattern = re.compile(r"/[a-zA-Z0-9_]*")
        self.word_completer = WordCompleter(words, pattern=self.pattern)

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        """Yield slash-command completions when the prompt starts with `/`."""
        if document.text_before_cursor.startswith("/"):
            yield from self.word_completer.get_completions(document, complete_event)


async def run_cli(args: Namespace) -> None:
    """Run the interactive CLI entry point."""
    config = build_default_agent_config(args)
    prompt_session = _create_prompt_session()

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
        output_task = asyncio.create_task(run_session_output(session=session, system_message=system_message))
        try:
            await _run_prompt_loop(session=session, prompt_session=prompt_session)
        finally:
            await session.close()
            output_task.cancel()
            await asyncio.gather(output_task, return_exceptions=True)


def _create_prompt_session() -> PromptSession[str]:
    """Create the prompt-toolkit session used by the interactive CLI."""
    history_dir = get_app_cache_dir()
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "history"
    return PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=True,
    )


async def _prompt_with_session(
    prompt_session: PromptSession[str],
    *,
    session: AgentSession,
    words: list[str] | None = None,
) -> str:
    """Prompt for input while showing the live session status in a footer."""
    Console().bell()
    print(Rule(style="dim"))
    completer = SlashCompleter(words) if words else None
    return await prompt_session.prompt_async(
        "> ",
        completer=completer,
        complete_while_typing=True,
        bottom_toolbar=lambda: format_session_status(session.state),
        refresh_interval=0.1,
    )


async def _run_prompt_loop(*, session: AgentSession, prompt_session: PromptSession[str]) -> None:
    """Keep a local prompt open and translate answers into session actions."""
    with patch_stdout(raw=True):
        while True:
            answer = await _prompt_with_session(prompt_session, session=session, words=CLI_COMMAND_NAMES)
            if await _handle_prompt_submission(session=session, answer=answer):
                return


async def _handle_prompt_submission(*, session: AgentSession, answer: str) -> bool:
    """Handle one prompt line and return true when the CLI should exit."""
    stripped = answer.strip()
    if stripped == "/exit":
        return True
    if stripped == "/help":
        print(
            "Available commands:\n"
            "  /exit\n"
            "  /help\n"
            "  /compact\n"
            "  /image <path-or-url>\n"
            "  /priority <prompt>\n"
            "  /interrupt <prompt>"
        )
        return False
    if stripped == "/compact":
        return not await session.enqueue_prompt(
            "Immediately compact our conversation so far by using the `compact_conversation` tool."
        )
    if stripped.startswith("/image"):
        parts = stripped.split(maxsplit=1)
        if len(parts) < 2:
            print("/image requires a path or URL argument.")
            return False
        data_url = await get_image(parts[1])
        image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
        return not await session.enqueue_prompt(image_content)
    if stripped.startswith("/priority"):
        priority_prompt = _extract_command_argument(answer=answer, command="/priority")
        if priority_prompt is None:
            print("/priority requires prompt text.")
            return False
        return not await session.enqueue_prompt(priority_prompt, priority=True)
    if stripped.startswith("/interrupt"):
        interrupt_prompt = _extract_command_argument(answer=answer, command="/interrupt")
        if interrupt_prompt is None:
            print("/interrupt requires prompt text.")
            return False
        return not await session.interrupt_and_enqueue(interrupt_prompt)

    return not await session.enqueue_prompt(answer)


def _extract_command_argument(*, answer: str, command: str) -> str | None:
    """Return the trailing argument text for one slash command."""
    if not answer.startswith(command):
        return None
    argument = answer[len(command) :].strip()
    if not argument:
        return None
    return argument
