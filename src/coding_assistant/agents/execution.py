import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from collections.abc import Awaitable, Callable
from json import JSONDecodeError

from coding_assistant.agents.callbacks import AgentProgressCallbacks, AgentToolCallbacks
from coding_assistant.agents.history import append_assistant_message, append_tool_message, append_user_message
from coding_assistant.agents.interrupts import InterruptController
from coding_assistant.agents.parameters import format_parameters
from coding_assistant.agents.types import (
    AgentContext,
    AgentDescription,
    AgentOutput,
    AgentState,
    Completer,
    FinishTaskResult,
    CompactConversationResult,
    TextResult,
)
from coding_assistant.llm.adapters import execute_tool_call, get_tools
from coding_assistant.trace import trace_data
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

CHAT_START_MESSAGE_TEMPLATE = """
## General

- You are an agent named `{name}`.
- You are in chat mode.
  - Use tools only when they materially advance the work.
  - When you do not know what to do next, reply without any tool calls to return control to the user. 

## Parameters

Your client has provided the following parameters for your session:

{parameters}
""".strip()


def _create_start_message(desc: AgentDescription) -> str:
    parameters_str = format_parameters(desc.parameters)
    message = START_MESSAGE_TEMPLATE.format(
        name=desc.name,
        parameters=parameters_str,
    )

    return message


def _create_chat_start_message(desc: AgentDescription) -> str:
    parameters_str = format_parameters(desc.parameters)
    message = CHAT_START_MESSAGE_TEMPLATE.format(
        name=desc.name,
        parameters=parameters_str,
    )
    return message


def _handle_finish_task_result(result: FinishTaskResult, state: AgentState):
    state.output = AgentOutput(result=result.result, summary=result.summary)
    return "Agent output set."


def _handle_text_result(result: TextResult) -> str:
    return result.content


def _clear_history(state: AgentState):
    """Resets the history to the first message (start message)."""
    state.history = [state.history[0]]


def _handle_compact_conversation_result(
    result: CompactConversationResult,
    desc: AgentDescription,
    state: AgentState,
    agent_callbacks: AgentProgressCallbacks,
):
    _clear_history(state)

    append_user_message(
        state.history,
        agent_callbacks,
        desc.name,
        f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work.",
        force=True,
    )

    return "Conversation compacted and history reset."


async def handle_tool_call(
    tool_call,
    ctx: AgentContext,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    *,
    ui: UI,
) -> str:
    """Execute a single tool call and return result_summary."""
    desc = ctx.desc
    state = ctx.state
    function_name = tool_call.function.name
    if not function_name:
        raise RuntimeError(f"Tool call {tool_call.id} is missing function name.")

    args_str = tool_call.function.arguments

    try:
        function_args = json.loads(args_str)
    except JSONDecodeError as e:
        logger.error(
            f"[{desc.name}] [{tool_call.id}] Failed to parse tool '{function_name}' arguments as JSON: {e} | raw: {args_str}"
        )
        return f"Error: Tool call arguments `{args_str}` are not valid JSON: {e}"

    logger.debug(f"[{tool_call.id}] [{desc.name}] Calling tool '{function_name}' with arguments {function_args}")

    # Notify callbacks that tool is starting
    agent_callbacks.on_tool_start(desc.name, tool_call.id, function_name, function_args)

    try:
        if callback_result := await tool_callbacks.before_tool_execution(
            desc.name,
            tool_call.id,
            function_name,
            function_args,
            ui=ui,
        ):
            logger.info(f"[{tool_call.id}] [{desc.name}] Tool '{function_name}' execution was prevented via callback.")
            function_call_result = callback_result
        else:
            function_call_result = await execute_tool_call(function_name, function_args, desc.tools)
    except Exception as e:
        function_call_result = TextResult(content=f"Error executing tool: {e}")

    trace_data(
        f"tool_result_{function_name}.json",
        json.dumps(
            {
                "tool_call_id": tool_call.id,
                "function_name": function_name,
                "function_args": function_args,
                "result": function_call_result.to_dict(),
            },
            indent=2,
        ),
    )

    result_handlers = {
        FinishTaskResult: lambda r: _handle_finish_task_result(r, state),
        CompactConversationResult: lambda r: _handle_compact_conversation_result(r, desc, state, agent_callbacks),
        TextResult: lambda r: _handle_text_result(r),
    }

    tool_return_summary = result_handlers[type(function_call_result)](function_call_result)
    return tool_return_summary


async def handle_tool_calls(
    message,
    ctx: AgentContext,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    *,
    ui: UI,
    task_created_callback: Callable[[str, asyncio.Task], None] | None = None,
):
    tool_calls = message.tool_calls

    if not tool_calls:
        return

    tasks_with_calls = []
    loop = asyncio.get_running_loop()
    for tool_call in tool_calls:
        task = loop.create_task(
            handle_tool_call(
                tool_call,
                ctx,
                agent_callbacks,
                tool_callbacks,
                ui=ui,
            ),
            name=f"{tool_call.function.name} ({tool_call.id})",
        )
        if task_created_callback is not None:
            task_created_callback(tool_call.id, task)
        tasks_with_calls.append((tool_call, task))

    done, pending = await asyncio.wait([task for _, task in tasks_with_calls])
    assert len(pending) == 0

    # Process results and append tool messages
    any_cancelled = False
    for tool_call, task in tasks_with_calls:
        try:
            result_summary = await task
        except asyncio.CancelledError:
            # Tool was cancelled
            result_summary = "Tool execution was cancelled."
            any_cancelled = True

        # Parse arguments from tool_call
        try:
            function_args = json.loads(tool_call.function.arguments)
        except JSONDecodeError:
            function_args = {}

        append_tool_message(
            ctx.state.history,
            agent_callbacks,
            ctx.desc.name,
            tool_call.id,
            tool_call.function.name,
            function_args,
            result_summary,
        )

    # Re-raise CancelledError if any tool was cancelled
    if any_cancelled:
        raise asyncio.CancelledError()


async def do_single_step(
    ctx: AgentContext,
    agent_callbacks: AgentProgressCallbacks,
    *,
    completer: Completer,
):
    desc = ctx.desc
    state = ctx.state

    tools = await get_tools(desc.tools)

    if not state.history:
        raise RuntimeError("Agent needs to have history in order to run a step.")

    completion = await completer(
        state.history,
        model=desc.model,
        tools=tools,
        callbacks=agent_callbacks,
    )
    message = completion.message

    if hasattr(message, "reasoning_content") and message.reasoning_content:
        agent_callbacks.on_assistant_reasoning(desc.name, message.reasoning_content)

    return message, completion.tokens


async def run_agent_loop(
    ctx: AgentContext,
    *,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    completer: Completer,
    ui: UI,
    compact_conversation_at_tokens: int = 200_000,
):
    desc = ctx.desc
    state = ctx.state

    if state.output is not None:
        raise RuntimeError("Agent already has a result or summary.")

    # Validate tools required for the agent loop
    if not any(tool.name() == "finish_task" for tool in desc.tools):
        raise RuntimeError("Agent needs to have a `finish_task` tool in order to run.")
    if not any(tool.name() == "compact_conversation" for tool in desc.tools):
        raise RuntimeError("Agent needs to have a `compact_conversation` tool in order to run.")

    start_message = _create_start_message(desc)
    agent_callbacks.on_agent_start(desc.name, desc.model, is_resuming=bool(state.history))
    append_user_message(state.history, agent_callbacks, desc.name, start_message)

    while state.output is None:
        message, tokens = await do_single_step(
            ctx,
            agent_callbacks,
            completer=completer,
        )

        # Append assistant message to history
        append_assistant_message(state.history, agent_callbacks, desc.name, message)

        if getattr(message, "tool_calls", []):
            await handle_tool_calls(
                message,
                ctx,
                agent_callbacks,
                tool_callbacks,
                ui=ui,
            )
        else:
            # Handle assistant steps without tool calls: inject corrective message
            append_user_message(
                state.history,
                agent_callbacks,
                desc.name,
                "I detected a step from you without any tool calls. This is not allowed. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
            )
        if tokens > compact_conversation_at_tokens:
            append_user_message(
                state.history,
                agent_callbacks,
                desc.name,
                "Your conversation history has grown too large. Compact it immediately by using the `compact_conversation` tool.",
            )

    assert state.output is not None

    agent_callbacks.on_agent_end(desc.name, state.output.result, state.output.summary)


class ChatCommandResult(Enum):
    PROCEED_WITH_MODEL = 1
    PROCEED_WITH_PROMPT = 2
    EXIT = 3


@dataclass
class ChatCommand:
    name: str
    help: str
    execute: Callable[[], Awaitable[ChatCommandResult]]


async def run_chat_loop(
    ctx: AgentContext,
    *,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    completer: Completer,
    ui: UI,
):
    desc = ctx.desc
    state = ctx.state

    if state.history:
        for message in state.history:
            if message.get("role") == "assistant":
                if content := message.get("content"):
                    agent_callbacks.on_assistant_message(desc.name, content, force=True)
            elif message.get("role") == "user":
                if content := message.get("content"):
                    agent_callbacks.on_user_message(desc.name, content, force=True)

    need_user_input = True

    async def _exit_cmd():
        return ChatCommandResult.EXIT

    async def _compact_cmd():
        append_user_message(
            state.history,
            agent_callbacks,
            desc.name,
            "Immediately compact our conversation so far by using the `compact_conversation` tool.",
            force=True,
        )

        nonlocal need_user_input
        need_user_input = True

        return ChatCommandResult.PROCEED_WITH_MODEL

    async def _clear_cmd():
        _clear_history(state)
        print("History cleared.")
        return ChatCommandResult.PROCEED_WITH_PROMPT

    commands = [
        ChatCommand("/exit", "Exit the chat", _exit_cmd),
        ChatCommand("/compact", "Compact the conversation history", _compact_cmd),
        ChatCommand("/clear", "Clear the conversation history", _clear_cmd),
    ]
    command_map = {cmd.name: cmd for cmd in commands}
    command_names = list(command_map.keys())

    start_message = _create_chat_start_message(desc)
    agent_callbacks.on_agent_start(desc.name, desc.model, is_resuming=bool(state.history))
    append_user_message(state.history, agent_callbacks, desc.name, start_message)

    while True:
        if need_user_input:
            need_user_input = False

            print()
            answer = await ui.prompt(words=command_names)
            answer_strip = answer.strip()

            if tool := command_map.get(answer_strip):
                result = await tool.execute()
                if result == ChatCommandResult.EXIT:
                    break
                elif result == ChatCommandResult.PROCEED_WITH_PROMPT:
                    need_user_input = True
                    continue
                elif result == ChatCommandResult.PROCEED_WITH_MODEL:
                    pass
            else:
                append_user_message(state.history, agent_callbacks, desc.name, answer)

        loop = asyncio.get_running_loop()
        with InterruptController(loop) as interrupt_controller:
            try:
                do_single_step_task = loop.create_task(
                    do_single_step(
                        ctx,
                        agent_callbacks,
                        completer=completer,
                    ),
                    name="do_single_step",
                )
                interrupt_controller.register_task("do_single_step", do_single_step_task)

                message, _ = await do_single_step_task
                append_assistant_message(state.history, agent_callbacks, desc.name, message)

                if getattr(message, "tool_calls", []):
                    await handle_tool_calls(
                        message,
                        ctx,
                        agent_callbacks,
                        tool_callbacks,
                        ui=ui,
                        task_created_callback=interrupt_controller.register_task,
                    )
                else:
                    need_user_input = True
            except asyncio.CancelledError:
                need_user_input = True
