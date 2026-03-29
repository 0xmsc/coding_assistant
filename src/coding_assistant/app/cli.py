from __future__ import annotations

import re
from argparse import Namespace
from collections.abc import Awaitable, Callable, Iterable

from coding_assistant.app.default_agent import (
    build_default_agent_config,
    build_initial_system_message,
    create_default_agent,
)
from coding_assistant.app.image import get_image
from coding_assistant.app.output import DeltaRenderer, print_system_message, print_tool_calls
from coding_assistant.core.agent import run_agent_event_stream
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.tool_calls import execute_tool_calls
from coding_assistant.infra.paths import get_app_cache_dir
from coding_assistant.integrations.mcp_client import print_mcp_tools
from coding_assistant.llm.types import BaseMessage, ContentDeltaEvent, Tool, UserMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, WordCompleter
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich import print
from rich.console import Console
from rich.rule import Rule

CLI_COMMAND_NAMES = ["/exit", "/help", "/compact", "/image"]


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

    async def prompt_user(words: list[str] | None = None) -> str:
        return await _prompt_with_session(prompt_session, words=words)

    async with create_default_agent(config=config) as bundle:
        if args.print_mcp_tools:
            await print_mcp_tools(bundle.mcp_servers)
            return

        system_message = build_initial_system_message(instructions=bundle.instructions)
        print_system_message(system_message)
        current_history: list[BaseMessage] = [system_message]

        await _drive_agent(
            history=current_history,
            model=args.model,
            tools=bundle.tools,
            prompt_user=prompt_user,
        )


async def _drive_agent(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    prompt_user: Callable[[list[str] | None], Awaitable[str]],
) -> None:
    """Drive the interactive agent loop until the CLI should exit."""
    current_history = list(history)
    while True:
        renderer = DeltaRenderer()
        boundary: AwaitingUser | AwaitingToolCalls | None = None
        async for event in run_agent_event_stream(
            history=current_history,
            model=model,
            tools=tools,
        ):
            if isinstance(event, ContentDeltaEvent):
                renderer.on_delta(event.content)
                continue
            if isinstance(event, (AwaitingUser, AwaitingToolCalls)):
                boundary = event

        if boundary is None:
            renderer.finish()
            raise RuntimeError("run_agent_event_stream stopped without yielding a boundary.")

        if isinstance(boundary, AwaitingToolCalls):
            renderer.finish(trailing_blank_line=False)
            print_tool_calls(boundary.message)
            current_history = await execute_tool_calls(
                boundary=boundary,
                tools=tools,
            )
            continue

        renderer.finish()
        current_history = boundary.history
        while True:
            answer = await prompt_user(CLI_COMMAND_NAMES)
            stripped = answer.strip()
            if stripped == "/exit":
                return
            if stripped == "/help":
                print("Available commands:\n  /exit\n  /help\n  /compact\n  /image <path-or-url>")
                continue
            if stripped == "/compact":
                current_history.append(
                    UserMessage(
                        content="Immediately compact our conversation so far by using the `compact_conversation` tool."
                    )
                )
                break
            if stripped.startswith("/image"):
                parts = stripped.split(maxsplit=1)
                if len(parts) < 2:
                    print("/image requires a path or URL argument.")
                    continue
                data_url = await get_image(parts[1])
                image_content = [{"type": "image_url", "image_url": {"url": data_url}}]
                current_history.append(UserMessage(content=image_content))
                break

            current_history.append(UserMessage(content=answer))
            break


def _create_prompt_session() -> PromptSession[str]:
    """Create the prompt-toolkit session used by the interactive CLI."""
    history_dir = get_app_cache_dir()
    history_dir.mkdir(parents=True, exist_ok=True)
    history_file = history_dir / "history"
    return PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=True,
    )


async def _prompt_with_session(prompt_session: PromptSession[str], *, words: list[str] | None = None) -> str:
    """Prompt for input with optional slash-command completion."""
    Console().bell()
    print(Rule(style="dim"))
    completer = SlashCompleter(words) if words else None
    return await prompt_session.prompt_async("> ", completer=completer, complete_while_typing=True)
