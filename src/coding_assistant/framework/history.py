from coding_assistant.framework.callbacks import ProgressCallbacks
from coding_assistant.llm.types import AssistantMessage, BaseMessage, ToolMessage, UserMessage


def append_tool_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    tool_call_id: str,
    function_name: str,
    function_args: dict,
    function_call_result: str,
):
    callbacks.on_tool_message(context_name, tool_call_id, function_name, function_args, function_call_result)

    history.append(
        ToolMessage(
            tool_call_id=tool_call_id,
            name=function_name,
            content=function_call_result,
        )
    )


def append_user_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    content: str,
    force: bool = False,
):
    callbacks.on_user_message(context_name, content, force=force)

    history.append(
        UserMessage(
            content=content,
        )
    )


def append_assistant_message(
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
    message: AssistantMessage,
    force: bool = False,
):
    if message.content:
        callbacks.on_assistant_message(context_name, message.content, force=force)

    history.append(message)


def clear_history(history: list[BaseMessage]):
    """Resets the history to the first message (the start message) in-place."""
    if history:
        history[:] = [history[0]]
