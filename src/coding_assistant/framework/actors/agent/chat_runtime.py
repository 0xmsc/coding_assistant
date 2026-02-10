from typing import TYPE_CHECKING

from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.actors.common.messages import ConfigureToolSetRequest
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import (
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
)
from coding_assistant.ui import UI

if TYPE_CHECKING:
    from coding_assistant.framework.execution import AgentActor, ToolCallActor


async def run_chat_loop(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    instructions: str | None,
    callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),  # Kept for API compatibility.
    completer: Completer,
    ui: UI,  # Kept for API compatibility.
    context_name: str,
    agent_actor: "AgentActor",
    tool_call_actor: "ToolCallActor",
    user_actor: UI,
) -> None:
    tools_with_meta = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversationTool())
    await tool_call_actor.send_message(ConfigureToolSetRequest(tools=tuple(tools_with_meta)))

    await agent_actor.run_chat_loop(
        history=history,
        model=model,
        tools=tools_with_meta,
        instructions=instructions,
        callbacks=callbacks,
        completer=completer,
        context_name=context_name,
        user_actor=user_actor,
        tool_call_actor=tool_call_actor,
    )
