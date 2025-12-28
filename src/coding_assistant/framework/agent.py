import logging


from coding_assistant.framework.builtin_tools import (
    CompactConversationTool,
    FinishTaskTool,
)
from coding_assistant.framework.callbacks import ProgressCallbacks, ToolCallbacks
from coding_assistant.framework.execution import do_single_step, handle_tool_calls
from coding_assistant.framework.history import (
    append_assistant_message,
    append_user_message,
    clear_history,
)
from coding_assistant.framework.parameters import format_parameters
from coding_assistant.framework.types import (
    AgentContext,
    AgentDescription,
    AgentOutput,
    AgentState,
    Completer,
)
from coding_assistant.framework.results import (
    CompactConversationResult,
    FinishTaskResult,
    TextResult,
    ToolResult,
)
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)

START_MESSAGE_TEMPLATE = """
## General

- You are an agent named `{name}`.
- You are given a set of parameters by your client, among which are your task and your description.
  - It is of the utmost importance that you try your best to fulfill the task as specified.
  - The task shall be done in a way which fits your description.
- You must use at least one tool call in every step.
  - Use the `finish_task` tool when you have fully finished your task, no questions should still be open.

## Parameters

Your client has provided the following parameters for your task:

{parameters}
""".strip()


def _create_start_message(desc: AgentDescription) -> str:
    parameters_str = format_parameters(desc.parameters)
    message = START_MESSAGE_TEMPLATE.format(
        name=desc.name,
        parameters=parameters_str,
    )

    return message


def _handle_finish_task_result(result: FinishTaskResult, state: AgentState):
    state.output = AgentOutput(result=result.result, summary=result.summary)
    return "Agent output set."


def _handle_compact_conversation_result(
    result: CompactConversationResult,
    desc: AgentDescription,
    state: AgentState,
    progress_callbacks: ProgressCallbacks,
):
    clear_history(state.history)

    append_user_message(
        state.history,
        progress_callbacks,
        desc.name,
        f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work.",
        force=True,
    )

    return "Conversation compacted and history reset."


def handle_tool_result_agent(
    result: ToolResult,
    desc: AgentDescription,
    state: AgentState,
    progress_callbacks: ProgressCallbacks,
) -> str:
    if isinstance(result, FinishTaskResult):
        return _handle_finish_task_result(result, state)
    if isinstance(result, CompactConversationResult):
        return _handle_compact_conversation_result(result, desc, state, progress_callbacks)
    if isinstance(result, TextResult):
        return result.content
    return f"Tool produced result of type {type(result).__name__}"


async def run_agent_loop(
    ctx: AgentContext,
    *,
    progress_callbacks: ProgressCallbacks,
    tool_callbacks: ToolCallbacks,
    completer: Completer,
    ui: UI,
    compact_conversation_at_tokens: int = 200_000,
):
    desc = ctx.desc
    state = ctx.state

    if state.output is not None:
        raise RuntimeError("Agent already has a result or summary.")

    tools = list(desc.tools)
    if not any(tool.name() == "finish_task" for tool in tools):
        tools.append(FinishTaskTool())
    if not any(tool.name() == "compact_conversation" for tool in tools):
        tools.append(CompactConversationTool())

    start_message = _create_start_message(desc)
    append_user_message(state.history, progress_callbacks, desc.name, start_message)

    while state.output is None:
        message, usage = await do_single_step(
            state.history,
            desc.model,
            tools,
            progress_callbacks,
            completer=completer,
            context_name=desc.name,
        )

        append_assistant_message(state.history, progress_callbacks, desc.name, message)

        if getattr(message, "tool_calls", []):
            await handle_tool_calls(
                message,
                tools,
                state.history,
                progress_callbacks,
                tool_callbacks,
                ui=ui,
                context_name=desc.name,
                handle_tool_result=lambda result: handle_tool_result_agent(result, desc, state, progress_callbacks),
            )
        else:
            append_user_message(
                state.history,
                progress_callbacks,
                desc.name,
                "I detected a step from you without any tool calls. This is not allowed. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
            )
        if usage is not None and usage.tokens > compact_conversation_at_tokens:
            append_user_message(
                state.history,
                progress_callbacks,
                desc.name,
                "Your conversation history has grown too large. Compact it immediately by using the `compact_conversation` tool.",
            )

    assert state.output is not None
