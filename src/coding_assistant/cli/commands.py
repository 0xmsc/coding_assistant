from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from rich import print as rich_print

from coding_assistant.cli.image import get_image
from coding_assistant.core.agent_session import AgentSession
from coding_assistant.core.compacting_session import COMPACTION_PROMPT, AutoCompactingSession


class SlashCommand(ABC):
    """Interface for an interactive CLI slash command."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The slash command name, e.g. '/exit'."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Short human-readable description of the command."""
        ...

    @property
    def usage(self) -> str:
        """Optional usage arguments, e.g. '<path-or-url>'."""
        return ""

    @abstractmethod
    async def execute(self, *, session: AgentSession | AutoCompactingSession, args: str) -> bool:
        """Execute the command.

        Returns:
            True if the application should exit, False otherwise.
        """
        ...


class ExitCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "/exit"

    @property
    def description(self) -> str:
        return "Exit the assistant"

    async def execute(self, *, session: AgentSession | AutoCompactingSession, args: str) -> bool:
        return True


class HelpCommand(SlashCommand):
    def __init__(self, commands_provider: Sequence[SlashCommand] | None = None) -> None:
        self._commands_provider = commands_provider

    @property
    def name(self) -> str:
        return "/help"

    @property
    def description(self) -> str:
        return "Show available commands"

    async def execute(self, *, session: AgentSession | AutoCompactingSession, args: str) -> bool:
        commands = self._commands_provider if self._commands_provider is not None else get_default_commands()
        lines = ["Available commands:"]
        for cmd in commands:
            cmd_usage = f" {cmd.usage}" if cmd.usage else ""
            lines.append(f"  {cmd.name}{cmd_usage} - {cmd.description}")
        rich_print("\n".join(lines))
        return False


class CompactCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "/compact"

    @property
    def description(self) -> str:
        return "Immediately compact conversation history"

    async def execute(self, *, session: AgentSession | AutoCompactingSession, args: str) -> bool:
        return not await session.enqueue_prompt(COMPACTION_PROMPT)


class ImageCommand(SlashCommand):
    @property
    def name(self) -> str:
        return "/image"

    @property
    def description(self) -> str:
        return "Attach an image to the prompt"

    @property
    def usage(self) -> str:
        return "<path-or-url>"

    async def execute(self, *, session: AgentSession | AutoCompactingSession, args: str) -> bool:
        stripped_args = args.strip()
        if not stripped_args:
            rich_print("[red]/image requires a path or URL[/red]")
            return False
        try:
            data_url = await get_image(stripped_args)
        except Exception as exc:
            rich_print(f"[red]Failed to load image: {exc}[/red]")
            return False
        image_content: list[dict[str, Any]] = [{"type": "image_url", "image_url": {"url": data_url}}]
        return not await session.enqueue_prompt(image_content)


def get_default_commands() -> list[SlashCommand]:
    """Return the list of standard built-in CLI slash commands."""
    commands: list[SlashCommand] = [
        ExitCommand(),
        CompactCommand(),
        ImageCommand(),
    ]
    help_cmd = HelpCommand(commands_provider=commands)
    commands.insert(1, help_cmd)
    return commands


async def execute_slash_command(
    *,
    commands: Sequence[SlashCommand],
    content: str,
    session: AgentSession | AutoCompactingSession,
) -> bool | None:
    """If content is a slash command, execute it and return the exit boolean; otherwise return None."""
    stripped = content.strip()
    if not stripped.startswith("/"):
        return None

    parts = stripped.split(maxsplit=1)
    command_name = parts[0]
    args = parts[1] if len(parts) > 1 else ""

    for cmd in commands:
        if cmd.name == command_name:
            return await cmd.execute(session=session, args=args)

    rich_print(f"[red]Unknown command: {command_name}. Type /help for available commands.[/red]")
    return False
