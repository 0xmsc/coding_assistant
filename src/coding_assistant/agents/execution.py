import asyncio
import dataclasses
import json
import logging
import re
import textwrap
from json import JSONDecodeError

from opentelemetry import trace

from coding_assistant.agents.callbacks import AgentProgressCallbacks, AgentToolCallbacks, NullToolCallbacks
from coding_assistant.agents.history import append_assistant_message, append_tool_message, append_user_message
from coding_assistant.agents.interrupts import InterruptibleSection, NonInterruptibleSection
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
You are an agent named `{name}`.

## Parameters

Your client has provided the following parameters for your task:

{parameters}
""".strip()

FEEDBACK_TEMPLATE = """
Your client has provided the following feedback on your work:

{feedback}

Please rework your result to address the feedback.
Afterwards, call the `finish_task` tool again to signal that you are done.
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
):
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
        append_tool_message(
            state.history,
            agent_callbacks,
            desc.name,
            tool_call.id,
            function_name,
            None,
            f"Error: Tool call arguments `{args_str}` are not valid JSON: {e}",
        )
        return

    trace.get_current_span().set_attribute("function.name", function_name)
    trace.get_current_span().set_attribute("function.args", json.dumps(function_args))

    logger.info(f"[{tool_call.id}] [{desc.name}] Calling tool '{function_name}' with arguments {function_args}")

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
        append_tool_message(
            state.history,
            agent_callbacks,
            desc.name,
            tool_call.id,
            function_name,
            function_args,
            f"Error executing tool: {e}",
        )
        return

    trace.get_current_span().set_attribute("function.result", str(function_call_result))

    result_handlers = {
        FinishTaskResult: lambda r: _handle_finish_task_result(r, state),
        ShortenConversationResult: lambda r: _handle_shorten_conversation_result(r, desc, state, agent_callbacks),
        TextResult: lambda r: _handle_text_result(r),
    }

    tool_return_summary = result_handlers[type(function_call_result)](function_call_result)

    append_tool_message(
        state.history,
        agent_callbacks,
        desc.name,
        tool_call.id,
        function_name,
        function_args,
        tool_return_summary,
    )


@tracer.start_as_current_span("handle_tool_calls")
async def handle_tool_calls(
    message,
    ctx: AgentContext,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    *,
    ui: UI,
):
    desc = ctx.desc
    state = ctx.state
    tool_calls = message.tool_calls
    if not tool_calls:
        append_user_message(
            state.history,
            agent_callbacks,
            desc.name,
            "I detected a step from you without any tool calls. This is not allowed. If you want to ask the client something, please use the `ask_user` tool. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
        )
        return

    trace.get_current_span().set_attribute("message.tool_calls", [x.model_dump_json() for x in tool_calls])

    aws = []
    for tool_call in tool_calls:
        task = asyncio.create_task(
            handle_tool_call(
                tool_call,
                ctx,
                agent_callbacks,
                tool_callbacks,
                ui=ui,
            ),
            name=f"{tool_call.function.name} ({tool_call.id})",
        )
        aws.append(task)

    while True:
        done, pending = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED, timeout=0.2)

        # await the task, which will throw any exceptions stored in the future.
        for task in done:
            await task

        if not pending:
            break


@tracer.start_as_current_span("do_single_step")
async def do_single_step(
    ctx: AgentContext,
    agent_callbacks: AgentProgressCallbacks,
    shorten_conversation_at_tokens: int,
    *,
    completer: Completer,
    ui: UI,
    tool_callbacks: AgentToolCallbacks,
):
    desc = ctx.desc
    state = ctx.state
    trace.get_current_span().set_attribute("agent.name", desc.name)

    # Validate agent tools
    if not any(tool.name() == "finish_task" for tool in desc.tools):
        raise RuntimeError("Agent needs to have a `finish_task` tool in order to run a step.")
    if not any(tool.name() == "shorten_conversation" for tool in desc.tools):
        raise RuntimeError("Agent needs to have a `shorten_conversation` tool in order to run a step.")

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

        # We delete reasoning so we don't store it in the history, and hence do not send it to the LLM again.
        del message.reasoning_content

    append_assistant_message(state.history, agent_callbacks, desc.name, message)
    await handle_tool_calls(
        message,
        ctx,
        agent_callbacks,
        tool_callbacks,
        ui=ui,
    )

    # Check conversation length and request shortening if needed
    if completion.tokens > shorten_conversation_at_tokens:
        append_user_message(
            state.history,
            agent_callbacks,
            desc.name,
            "Your conversation history has grown too large. Please summarize it by using the `shorten_conversation` tool.",
        )

    return message


@tracer.start_as_current_span("run_agent_loop")
async def run_agent_loop(
    ctx: AgentContext,
    *,
    agent_callbacks: AgentProgressCallbacks,
    tool_callbacks: AgentToolCallbacks,
    completer: Completer,
    ui: UI,
    shorten_conversation_at_tokens: int = 200_000,
    enable_user_feedback: bool = False,
    is_interruptible: bool = False,
):
    desc = ctx.desc
    state = ctx.state

    if state.output is not None:
        raise RuntimeError("Agent already has a result or summary.")

    trace.get_current_span().set_attribute("agent.name", desc.name)
    parameters_json = json.dumps([dataclasses.asdict(p) for p in desc.parameters])
    trace.get_current_span().set_attribute("agent.parameter_description", parameters_json)

    start_message = _create_start_message(desc)
    agent_callbacks.on_agent_start(desc.name, desc.model, is_resuming=bool(state.history))
    append_user_message(state.history, agent_callbacks, desc.name, start_message)

    while True:
        while state.output is None:
            section_cls = InterruptibleSection if is_interruptible else NonInterruptibleSection
            with section_cls() as interruptible_section:
                await do_single_step(
                    ctx,
                    agent_callbacks,
                    shorten_conversation_at_tokens,
                    completer=completer,
                    ui=ui,
                    tool_callbacks=tool_callbacks,
                )

            if interruptible_section.was_interrupted:
                logger.info(f"Agent '{desc.name}' was interrupted during execution.")
                feedback = await ui.ask("Feedback: ")
                formatted_feedback = FEEDBACK_TEMPLATE.format(
                    feedback=textwrap.indent(feedback, "> "),
                )
                append_user_message(state.history, agent_callbacks, desc.name, formatted_feedback)

        assert state.output is not None, "Agent did not produce output"

        trace.get_current_span().set_attribute("agent.result", state.output.result)
        trace.get_current_span().set_attribute("agent.summary", state.output.summary)

        agent_callbacks.on_agent_end(desc.name, state.output.result, state.output.summary)

        user_feedback: str = "Ok"
        if enable_user_feedback:
            user_feedback = await ui.ask(f"Feedback for {desc.name}", default="Ok")

        if user_feedback != "Ok":
            formatted_feedback = FEEDBACK_TEMPLATE.format(feedback=textwrap.indent(user_feedback, "> "))
            append_user_message(state.history, agent_callbacks, desc.name, formatted_feedback)
            state.output = None  # continue loop
        else:
            break

    assert state.output is not None
