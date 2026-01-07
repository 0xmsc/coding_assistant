from typing import Any
from coding_assistant.framework.callbacks import ProgressCallbacks
from coding_assistant.llm.types import AssistantMessage, BaseMessage, ToolMessage, UserMessage


def append_tool_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    message: ToolMessage,
    arguments: dict[str, Any],
) -> None:
    callbacks.on_tool_message(context_name, message, message.name or "", arguments)
    history.append(message)


def append_user_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    message: UserMessage,
    force: bool = False,
) -> None:
    callbacks.on_user_message(context_name, message, force=force)
    history.append(message)


def append_assistant_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    message: AssistantMessage,
    force: bool = False,
) -> None:
    if message.content:
        callbacks.on_assistant_message(context_name, message, force=force)

    history.append(message)


def clear_history(history: list[BaseMessage]) -> None:
    """Resets the history to the first message (the start message) in-place."""
    if history:
        history[:] = [history[0]]
