from coding_assistant.framework.builtin_tools import (
    CompactConversationTool,
    FinishTaskTool,
)
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks
from coding_assistant.framework.execution import AgentActor, ToolCallActor
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState, Completer
from coding_assistant.framework.results import CompactConversationResult, FinishTaskResult
from coding_assistant.llm.types import ToolResult
from coding_assistant.ui import ActorUI, UI, UserActor


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
    agent_actor: AgentActor | None = None,
    tool_call_actor: ToolCallActor | None = None,
    user_actor: UI | None = None,
) -> None:
    tools_with_meta = list(ctx.desc.tools)
    if not any(tool.name() == "finish_task" for tool in tools_with_meta):
        tools_with_meta.append(FinishTaskTool())
    if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
        tools_with_meta.append(CompactConversationTool())

    owns_user_actor = user_actor is None
    resolved_user_actor = user_actor
    if resolved_user_actor is None:
        resolved_user_actor = ui if isinstance(ui, ActorUI) else UserActor(ui, context_name=ctx.desc.name)
    owns_tool_call_actor = tool_call_actor is None
    resolved_tool_call_actor = tool_call_actor or ToolCallActor(
        tools=tools_with_meta,
        ui=resolved_user_actor,
        context_name=ctx.desc.name,
        progress_callbacks=progress_callbacks,
        tool_callbacks=tool_callbacks,
    )
    owns_agent_actor = agent_actor is None
    resolved_agent_actor = agent_actor or AgentActor(context_name=ctx.desc.name)

    if owns_user_actor and isinstance(resolved_user_actor, ActorUI):
        resolved_user_actor.start()
    if owns_tool_call_actor:
        resolved_tool_call_actor.start()
    if owns_agent_actor:
        resolved_agent_actor.start()

    try:
        await resolved_agent_actor.run_agent_loop(
            ctx,
            tools=tools_with_meta,
            progress_callbacks=progress_callbacks,
            completer=completer,
            compact_conversation_at_tokens=compact_conversation_at_tokens,
            tool_call_actor=resolved_tool_call_actor,
        )
    finally:
        ctx.state.history = await resolved_agent_actor.get_agent_history(id(ctx.state))
        if owns_tool_call_actor:
            await resolved_tool_call_actor.stop()
        if owns_agent_actor:
            await resolved_agent_actor.stop()
        if owns_user_actor and isinstance(resolved_user_actor, ActorUI):
            await resolved_user_actor.stop()
