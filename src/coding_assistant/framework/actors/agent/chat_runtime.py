from typing import TYPE_CHECKING

from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import (
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
)

if TYPE_CHECKING:
    from coding_assistant.framework.execution import AgentActor


async def run_chat_loop(
    *,
    history: list[BaseMessage],
    model: str,
    tools: list[Tool],
    instructions: str | None,
    callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),  # Kept for API compatibility.
    completer: Completer,
    ui: object,  # Kept for API compatibility.
    context_name: str,
    agent_actor: "AgentActor",
    tool_call_actor_uri: str,
    user_actor_uri: str,
) -> None:
    tools_with_meta = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversationTool())

    await agent_actor.run_chat_loop(
        history=history,
        model=model,
        tools=tools_with_meta,
        instructions=instructions,
        callbacks=callbacks,
        completer=completer,
        context_name=context_name,
        user_actor_uri=user_actor_uri,
        tool_call_actor_uri=tool_call_actor_uri,
    )
