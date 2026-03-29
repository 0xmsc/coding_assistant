from __future__ import annotations

from collections.abc import Sequence

from coding_assistant.llm.types import BaseMessage, SystemMessage, UserMessage


SYSTEM_PROMPT_TEMPLATE = """
## General

- You are an agent.
- Use tools when they materially advance the work.
- When you want the client to reply, write a normal assistant message without tool calls.

{instructions_section}
""".strip()


def _render_instructions_section(instructions: str) -> str:
    """Render the optional instructions section only when it has content."""
    cleaned = instructions.strip()
    if not cleaned:
        return ""
    return f"## Instructions\n\n{cleaned}"


def build_system_prompt(*, instructions: str) -> str:
    """Render the top-level system prompt for a new transcript."""
    return SYSTEM_PROMPT_TEMPLATE.format(instructions_section=_render_instructions_section(instructions))


def compact_history(history: Sequence[BaseMessage], summary: str) -> list[BaseMessage]:
    """Replace prior turns with a summary while keeping the original system prompt."""
    if not history:
        raise RuntimeError("History is empty.")

    first_message = history[0]
    if not isinstance(first_message, SystemMessage):
        raise RuntimeError("Cannot compact history without an initial system message.")

    return [
        first_message,
        UserMessage(content=f"A summary of your conversation until now:\n\n{summary}\n\nPlease continue your work."),
    ]
