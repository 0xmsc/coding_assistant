from __future__ import annotations

from coding_assistant.llm.types import SystemMessage


def build_initial_system_message(*, instructions: str) -> SystemMessage:
    """Build the system message used to seed a fresh transcript."""
    return SystemMessage(content=instructions)
