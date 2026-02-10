from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
import logging

from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import (
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
    ToolResult,
    UserMessage,
)
from typing import TYPE_CHECKING
from coding_assistant.framework.history import (
    append_user_message,
    clear_history,
)
from coding_assistant.framework.types import Completer
from coding_assistant.framework.results import CompactConversationResult, TextResult
from coding_assistant.ui import ActorUI, UI, UserActor

if TYPE_CHECKING:
    from coding_assistant.framework.execution import AgentActor, ToolCallActor

logger = logging.getLogger(__name__)

CHAT_START_MESSAGE_TEMPLATE = """
## General

- You are an agent.
- You are in chat mode.
  - Use tools only when they materially advance the work.
  - When you have finished your task, reply without any tool calls to return control to the user.
  - When you want to ask the user a question, create a message without any tool calls to return control to the user.

{instructions_section}
""".strip()


def _create_chat_start_message(instructions: str | None) -> str:
    instructions_section = ""
    if instructions:
        instructions_section = f"## Instructions\n\n{instructions}"
    message = CHAT_START_MESSAGE_TEMPLATE.format(
        instructions_section=instructions_section,
    )
    return message


def handle_tool_result_chat(
    result: ToolResult,
    *,
    history: list[BaseMessage],
    callbacks: ProgressCallbacks,
    context_name: str,
) -> str:
    if isinstance(result, CompactConversationResult):
        clear_history(history)
        compact_result_msg = UserMessage(
            content=f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work."
        )
        append_user_message(
            history,
            callbacks=callbacks,
            context_name=context_name,
            message=compact_result_msg,
            force=False,
        )
        return "Conversation compacted and history reset."

    if isinstance(result, TextResult):
        return result.content

    raise RuntimeError(f"Tool produced unexpected result of type {type(result).__name__}")


class ChatCommandResult(Enum):
    PROCEED_WITH_MODEL = 1
    PROCEED_WITH_PROMPT = 2
    EXIT = 3


@dataclass
class ChatCommand:
    name: str
    help: str
    execute: Callable[[str | None], Awaitable[ChatCommandResult]]


async def run_chat_loop(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    instructions: str | None,
    callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    completer: Completer,
    ui: UI,
    context_name: str,
    agent_actor: "AgentActor | None" = None,
    tool_call_actor: "ToolCallActor | None" = None,
    user_actor: UI | None = None,
) -> None:
    tools_with_meta = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversationTool())
    owns_user_actor = user_actor is None
    resolved_user_actor = user_actor
    if resolved_user_actor is None:
        resolved_user_actor = ui if isinstance(ui, ActorUI) else UserActor(ui, context_name=context_name)
    owns_tool_call_actor = tool_call_actor is None
    resolved_tool_call_actor = tool_call_actor or _tool_call_actor(
        tools=tools_with_meta,
        ui=resolved_user_actor,
        context_name=context_name,
        progress_callbacks=callbacks,
        tool_callbacks=tool_callbacks,
    )
    owns_agent_actor = agent_actor is None
    resolved_agent_actor = agent_actor or _agent_actor(context_name=context_name)

    if owns_user_actor and isinstance(resolved_user_actor, ActorUI):
        resolved_user_actor.start()
    if owns_tool_call_actor:
        resolved_tool_call_actor.start()
    if owns_agent_actor:
        resolved_agent_actor.start()

    await resolved_agent_actor.set_history(history)
    try:
        await resolved_agent_actor.run_chat_loop(
            model=model,
            tools=tools_with_meta,
            instructions=instructions,
            callbacks=callbacks,
            completer=completer,
            context_name=context_name,
            user_actor=resolved_user_actor,
            tool_call_actor=resolved_tool_call_actor,
        )
    finally:
        history[:] = await resolved_agent_actor.get_history()
        if owns_tool_call_actor:
            await resolved_tool_call_actor.stop()
        if owns_agent_actor:
            await resolved_agent_actor.stop()
        if owns_user_actor and isinstance(resolved_user_actor, ActorUI):
            await resolved_user_actor.stop()


def _agent_actor(*, context_name: str) -> "AgentActor":
    from coding_assistant.framework.execution import AgentActor

    return AgentActor(context_name=context_name)


def _tool_call_actor(
    *,
    tools: list[Tool],
    ui: UI,
    context_name: str,
    progress_callbacks: ProgressCallbacks,
    tool_callbacks: ToolCallbacks,
) -> "ToolCallActor":
    from coding_assistant.framework.execution import ToolCallActor

    return ToolCallActor(
        tools=tools,
        ui=ui,
        context_name=context_name,
        progress_callbacks=progress_callbacks,
        tool_callbacks=tool_callbacks,
    )
