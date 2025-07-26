import dataclasses
import json
import logging
import sys
import textwrap

from opentelemetry import trace

from coding_assistant.agents.callbacks import AgentCallbacks
from coding_assistant.agents.history import append_assistant_message, append_tool_message, append_user_message
from coding_assistant.agents.parameters import format_parameters
from coding_assistant.agents.types import Agent, AgentOutput, FinishTaskResult, ShortenConversationResult, TextResult
from coding_assistant.llm.adapters import execute_tool_call, get_tools
from coding_assistant.llm.model import complete

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


START_MESSAGE_TEMPLATE = """
You are an agent named `{name}`.

## Task

Your client has been given the following description of your work and capabilities: 

{description}

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


def create_start_message(agent: Agent) -> str:
    """
    Compose the canonical agent start message, including all provided parameters in readable format.
    Uses formatting helpers from coding_assistant.agents.utils.
    """
    parameters_str = format_parameters(agent.parameters)
    return START_MESSAGE_TEMPLATE.format(
        name=agent.name,
        description=textwrap.indent(agent.description, "> "),
        parameters=parameters_str,
    )


def _handle_finish_task_result(result: FinishTaskResult, agent: Agent):
    agent.output = AgentOutput(
        result=result.result,
        summary=result.summary,
        feedback=result.feedback,
    )
    return "Agent output set."


def _handle_shorten_conversation_result(
    result: ShortenConversationResult, agent: Agent, agent_callbacks: AgentCallbacks
):
    start_message = create_start_message(agent)
    agent.history = []
    append_user_message(
        agent.history,
        agent_callbacks,
        agent.name,
        start_message,
    )
    append_user_message(
        agent.history,
        agent_callbacks,
        agent.name,
        f"A summary of your conversation with the client until now:\n\n{result.summary}\n\nPlease continue your work.",
    )
    return "Conversation shortened and history reset."


@tracer.start_as_current_span("handle_tool_call")
async def handle_tool_call(tool_call, agent: Agent, agent_callbacks: AgentCallbacks):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")

    trace.get_current_span().set_attribute("function.name", function_name)
    trace.get_current_span().set_attribute("function.args", tool_call.function.arguments)

    function_call_result = await execute_tool_call(function_name, function_args, agent.tools, agent.mcp_servers)

    assert function_call_result is not None, f"Function {function_name} not implemented"
    trace.get_current_span().set_attribute("function.result", str(function_call_result))

    result_handlers = {
        FinishTaskResult: lambda r: _handle_finish_task_result(r, agent),
        ShortenConversationResult: lambda r: _handle_shorten_conversation_result(r, agent, agent_callbacks),
        TextResult: lambda r: r.content if len(r.content) <= 50_000 else "System error: Tool call result too long.",
    }

    handler = result_handlers.get(type(function_call_result))
    if handler:
        tool_return_summary = handler(function_call_result)
    else:
        raise TypeError(f"Unknown tool result type: {type(function_call_result)}")

    append_tool_message(
        agent.history, agent_callbacks, agent.name, tool_call.id, function_name, function_args, tool_return_summary
    )


def _validate_agent_tools(agent: Agent):
    if not any(tool.name() == "finish_task" for tool in agent.tools):
        raise RuntimeError("Agent needs to have a `finish_task` tool in order to run a step.")
    if not any(tool.name() == "shorten_conversation" for tool in agent.tools):
        raise RuntimeError("Agent needs to have a `shorten_conversation` tool in order to run a step.")


def _handle_no_tool_calls(agent: Agent, agent_callbacks: AgentCallbacks):
    append_user_message(
        agent.history,
        agent_callbacks,
        agent.name,
        "I detected a step from you without any tool calls. This is not allowed. If you want to ask the client something, please use the `ask_user` tool. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
    )


def _check_conversation_length(
    agent: Agent, agent_callbacks: AgentCallbacks, tokens: int, shorten_conversation_at_tokens: int
):
    if tokens > shorten_conversation_at_tokens:
        append_user_message(
            agent.history,
            agent_callbacks,
            agent.name,
            "Your conversation history has grown too large. Please summarize it by using the `shorten_conversation` tool.",
        )


@tracer.start_as_current_span("do_single_step")
async def do_single_step(agent: Agent, agent_callbacks: AgentCallbacks, shorten_conversation_at_tokens: int):
    trace.get_current_span().set_attribute("agent.name", agent.name)
    _validate_agent_tools(agent)

    tools = await get_tools(agent.tools, agent.mcp_servers)
    trace.get_current_span().set_attribute("agent.tools", json.dumps(tools))

    if not agent.history:
        raise RuntimeError("Agent needs to have history in order to run a step.")
    trace.get_current_span().set_attribute("agent.history", json.dumps(agent.history))

    completion = await complete(
        agent.history,
        model=agent.model,
        tools=tools,
        callbacks=agent_callbacks,
    )
    message = completion.message
    trace.get_current_span().set_attribute("completion.message", message.model_dump_json())

    if hasattr(message, "reasoning_content") and message.reasoning_content:
        trace.get_current_span().set_attribute("completion.reasoning_content", message.reasoning_content)
        del message.reasoning_content

    append_assistant_message(agent.history, agent_callbacks, agent.name, message)

    if message.tool_calls:
        for tool_call in message.tool_calls:
            await handle_tool_call(tool_call, agent, agent_callbacks)
    else:
        _handle_no_tool_calls(agent, agent_callbacks)

    _check_conversation_length(agent, agent_callbacks, completion.tokens, shorten_conversation_at_tokens)

    return message


@tracer.start_as_current_span("run_agent_loop")
async def run_agent_loop(
    agent: Agent,
    agent_callbacks: AgentCallbacks,
    shorten_conversation_at_tokens: int,
) -> AgentOutput:
    if agent.output:
        raise RuntimeError("Agent already has a result or summary.")

    trace.get_current_span().set_attribute("agent.name", agent.name)
    parameters_json = json.dumps([dataclasses.asdict(p) for p in agent.parameters])
    trace.get_current_span().set_attribute("agent.parameter_description", parameters_json)

    start_message = create_start_message(agent)
    agent_callbacks.on_agent_start(agent.name, agent.model, is_resuming=bool(agent.history))
    append_user_message(agent.history, agent_callbacks, agent.name, start_message)

    while True:
        while not agent.output:
            await do_single_step(agent, agent_callbacks, shorten_conversation_at_tokens)

        trace.get_current_span().set_attribute("agent.result", agent.output.result)
        trace.get_current_span().set_attribute("agent.summary", agent.output.summary)
        agent_callbacks.on_agent_end(agent.name, agent.output.result, agent.output.summary)

        feedback = await agent.feedback_function(agent)
        if feedback:
            formatted_feedback = FEEDBACK_TEMPLATE.format(
                feedback=textwrap.indent(feedback, "  "),
            )
            append_user_message(agent.history, agent_callbacks, agent.name, formatted_feedback)

            # Clear output so agent continues working
            agent.output = None
        else:
            # No feedback - we're done
            break

    return agent.output
