from typing import Any
from coding_assistant.framework.callbacks import ProgressCallbacks
from coding_assistant.llm.types import AssistantMessage, BaseMessage, ToolMessage, UserMessage


def append_tool_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    tool_call_id: str,
    function_name: str,
    function_args: dict[str, Any],
    function_call_result: str,
) -> None:
    message = ToolMessage(
        tool_call_id=tool_call_id,
        name=function_name,
        content=function_call_result,
    )
    callbacks.on_tool_message(context_name, message, function_name, function_args)

    history.append(message)


def append_user_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    content: str,
    force: bool = False,
) -> None:
    message = UserMessage(content=content)
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
