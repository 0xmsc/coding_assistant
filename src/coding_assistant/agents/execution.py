import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable
from json import JSONDecodeError

from opentelemetry import trace

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
    ShortenConversationResult,
    TextResult,
)
from coding_assistant.llm.adapters import execute_tool_call, get_tools
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


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
- You are in chat mode. You may converse without using tools. When you do not know what to do next, reply without any tool calls to return control to the user. Use tools only when they materially advance the work.

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


def _handle_shorten_conversation_result(
    result: ShortenConversationResult,
    desc: AgentDescription,
    state: AgentState,
    agent_callbacks: AgentProgressCallbacks,
):
    start_message = _create_start_message(desc)
    state.history = []
    append_user_message(
        state.history,
        agent_callbacks,
        desc.name,
        start_message,
    )
    append_user_message(
        state.history,
        agent_callbacks,
        desc.name,
        f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work.",
    )
    return "Conversation shortened and history reset."


@tracer.start_as_current_span("handle_tool_call")
async def handle_tool_call(
    tool_call,
    ctx: AgentContext,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    *,
    ui: UI,
) -> tuple[str, dict | None, str]:
    """Execute a single tool call and return (function_name, function_args, result_summary)."""
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
        return (function_name, None, f"Error: Tool call arguments `{args_str}` are not valid JSON: {e}")

    trace.get_current_span().set_attribute("function.name", function_name)
    trace.get_current_span().set_attribute("function.args", json.dumps(function_args))

    logger.debug(f"[{tool_call.id}] [{desc.name}] Calling tool '{function_name}' with arguments {function_args}")

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
    except ValueError as e:
        return (function_name, function_args, f"Error executing tool: {e}")

    trace.get_current_span().set_attribute("function.result", str(function_call_result))

    result_handlers = {
        FinishTaskResult: lambda r: _handle_finish_task_result(r, state),
        ShortenConversationResult: lambda r: _handle_shorten_conversation_result(r, desc, state, agent_callbacks),
        TextResult: lambda r: _handle_text_result(r),
    }

    tool_return_summary = result_handlers[type(function_call_result)](function_call_result)
    return (function_name, function_args, tool_return_summary)


@tracer.start_as_current_span("handle_tool_calls")
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

    trace.get_current_span().set_attribute("message.tool_calls", [x.model_dump_json() for x in tool_calls])

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
    for tool_call, task in tasks_with_calls:
        try:
            function_name, function_args, result_summary = await task
            append_tool_message(
                ctx.state.history,
                agent_callbacks,
                ctx.desc.name,
                tool_call.id,
                function_name,
                function_args,
                result_summary,
            )
        except asyncio.CancelledError:
            # Tool was cancelled - append a cancellation message
            append_tool_message(
                ctx.state.history,
                agent_callbacks,
                ctx.desc.name,
                tool_call.id,
                tool_call.function.name,
                None,
                "Tool execution was cancelled.",
            )


@tracer.start_as_current_span("do_single_step")
async def do_single_step(
    ctx: AgentContext,
    agent_callbacks: AgentProgressCallbacks,
    *,
    completer: Completer,
):
    desc = ctx.desc
    state = ctx.state
    trace.get_current_span().set_attribute("agent.name", desc.name)

    tools = await get_tools(desc.tools)
    trace.get_current_span().set_attribute("agent.tools", json.dumps(tools))

    if not state.history:
        raise RuntimeError("Agent needs to have history in order to run a step.")
    trace.get_current_span().set_attribute("agent.history", json.dumps(state.history))

    completion = await completer(
        state.history,
        model=desc.model,
        tools=tools,
        callbacks=agent_callbacks,
    )
    message = completion.message
    trace.get_current_span().set_attribute("completion.message", message.model_dump_json())

    if hasattr(message, "reasoning_content") and message.reasoning_content:
        trace.get_current_span().set_attribute("completion.reasoning_content", message.reasoning_content)
        agent_callbacks.on_assistant_reasoning(desc.name, message.reasoning_content)

    append_assistant_message(state.history, agent_callbacks, desc.name, message)

    return message, completion.tokens


@tracer.start_as_current_span("run_agent_loop")
async def run_agent_loop(
    ctx: AgentContext,
    *,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    completer: Completer,
    ui: UI,
    shorten_conversation_at_tokens: int = 200_000,
):
    desc = ctx.desc
    state = ctx.state

    if state.output is not None:
        raise RuntimeError("Agent already has a result or summary.")

    trace.get_current_span().set_attribute("agent.name", desc.name)
    parameters_json = json.dumps([dataclasses.asdict(p) for p in desc.parameters])
    trace.get_current_span().set_attribute("agent.parameter_description", parameters_json)

    # Validate tools required for the agent loop
    if not any(tool.name() == "finish_task" for tool in desc.tools):
        raise RuntimeError("Agent needs to have a `finish_task` tool in order to run.")
    if not any(tool.name() == "shorten_conversation" for tool in desc.tools):
        raise RuntimeError("Agent needs to have a `shorten_conversation` tool in order to run.")

    start_message = _create_start_message(desc)
    agent_callbacks.on_agent_start(desc.name, desc.model, is_resuming=bool(state.history))
    append_user_message(state.history, agent_callbacks, desc.name, start_message)

    while state.output is None:
        message, tokens = await do_single_step(
            ctx,
            agent_callbacks,
            completer=completer,
        )

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
        if tokens > shorten_conversation_at_tokens:
            append_user_message(
                state.history,
                agent_callbacks,
                desc.name,
                "Your conversation history has grown too large. Please summarize it by using the `shorten_conversation` tool.",
            )

    assert state.output is not None

    trace.get_current_span().set_attribute("agent.result", state.output.result)
    trace.get_current_span().set_attribute("agent.summary", state.output.summary)

    agent_callbacks.on_agent_end(desc.name, state.output.result, state.output.summary)


@tracer.start_as_current_span("run_chat_loop")
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

    trace.get_current_span().set_attribute("agent.name", desc.name)
    parameters_json = json.dumps([dataclasses.asdict(p) for p in desc.parameters])
    trace.get_current_span().set_attribute("agent.parameter_description", parameters_json)

    start_message = _create_chat_start_message(desc)
    agent_callbacks.on_agent_start(desc.name, desc.model, is_resuming=bool(state.history))
    append_user_message(state.history, agent_callbacks, desc.name, start_message)

    need_user_input = True
    loop = asyncio.get_running_loop()
    interrupt_controller = InterruptController(loop)

    while True:
        if need_user_input:
            answer = await ui.prompt()
            if answer.strip() == "/exit":
                break
            append_user_message(state.history, agent_callbacks, desc.name, answer)

        with interrupt_controller:
            try:
                message, _tokens = await do_single_step(
                    ctx,
                    agent_callbacks,
                    completer=completer,
                )

                if getattr(message, "tool_calls", []):
                    await handle_tool_calls(
                        message,
                        ctx,
                        agent_callbacks,
                        tool_callbacks,
                        ui=ui,
                        task_created_callback=interrupt_controller.register_task,
                    )
                    need_user_input = False
                else:
                    need_user_input = True
            except asyncio.CancelledError:
                if interrupt_controller.has_pending_interrupt:
                    interrupt_controller.consume_interrupts()
                    need_user_input = True
                    continue
                raise
