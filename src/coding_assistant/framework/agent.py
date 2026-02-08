from coding_assistant.framework.builtin_tools import (
    CompactConversationTool,
    FinishTaskTool,
)
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.framework.execution import AgentActor
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState, Completer
from coding_assistant.framework.results import CompactConversationResult, FinishTaskResult
from coding_assistant.llm.types import ToolResult
from coding_assistant.ui import UI
from coding_assistant.framework.system_actors import SystemActors, system_actor_scope


def _handle_finish_task_result(result: FinishTaskResult, *, state: AgentState) -> str:
    return AgentActor.handle_finish_task_result(result, state=state)


def _handle_compact_conversation_result(
    result: CompactConversationResult,
    *,
    desc: AgentDescription,
    state: AgentState,
    progress_callbacks: ProgressCallbacks,
) -> str:
    return AgentActor.handle_compact_conversation_result(
        result, desc=desc, state=state, progress_callbacks=progress_callbacks
    )


def handle_tool_result_agent(
    result: ToolResult,
    *,
    desc: AgentDescription,
    state: AgentState,
    progress_callbacks: ProgressCallbacks,
) -> str:
    return AgentActor.handle_tool_result_agent(result, desc=desc, state=state, progress_callbacks=progress_callbacks)


async def run_agent_loop(
    ctx: AgentContext,
    *,
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    completer: Completer,
    ui: UI,
    compact_conversation_at_tokens: int = 200_000,
    system_actors: SystemActors | None = None,
) -> None:
    tools_with_meta = list(ctx.desc.tools)
    if not any(tool.name() == "finish_task" for tool in tools_with_meta):
        tools_with_meta.append(FinishTaskTool())
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversationTool())

    if system_actors is None:
        async with system_actor_scope(
            tools=tools_with_meta,
            ui=ui,
            progress_callbacks=progress_callbacks,
            tool_callbacks=tool_callbacks,
            context_name=ctx.desc.name,
        ) as actors:
            await actors.run_agent_loop(
                ctx,
                tools=tools_with_meta,
                progress_callbacks=progress_callbacks,
                completer=completer,
                compact_conversation_at_tokens=compact_conversation_at_tokens,
            )
        return

    await system_actors.run_agent_loop(
        ctx,
        tools=tools_with_meta,
        progress_callbacks=progress_callbacks,
        completer=completer,
        compact_conversation_at_tokens=compact_conversation_at_tokens,
    )
