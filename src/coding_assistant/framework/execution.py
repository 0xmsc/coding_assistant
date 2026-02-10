from collections.abc import Sequence

from coding_assistant.framework.actors.agent.actor import AgentActor
from coding_assistant.framework.actors.llm.actor import LLMActor
from coding_assistant.framework.actors.tool_call.actor import ToolCallActor
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
    Usage,
)


async def do_single_step(
    *,
    history: list[BaseMessage],
    model: str,
    tools: Sequence[Tool],
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    completer: Completer,
    context_name: str,
    agent_actor: AgentActor,
) -> tuple[AssistantMessage, Usage | None]:
    return await agent_actor.do_single_step(
        history=history,
        model=model,
        tools=tools,
        progress_callbacks=progress_callbacks,
        completer=completer,
        context_name=context_name,
    )


__all__ = ["AgentActor", "LLMActor", "ToolCallActor", "do_single_step"]
