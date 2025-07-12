import dataclasses
import json
import logging
import textwrap

from opentelemetry import trace

from coding_assistant.agents.callbacks import AgentCallbacks
from coding_assistant.agents.history import append_tool_message, append_user_message, append_assistant_message
from coding_assistant.agents.types import Agent, AgentOutput, TextResult, FinishTaskResult, ShortenConversationResult
from coding_assistant.agents.parameters import format_parameters
from coding_assistant.llm.model import complete
from coding_assistant.llm.adapters import get_tools, execute_tool_call

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
        description=textwrap.indent(agent.description, "  "),
        parameters=textwrap.indent(parameters_str, "  "),
    )


@tracer.start_as_current_span("handle_tool_call")
async def handle_tool_call(tool_call, agent: Agent, agent_callbacks: AgentCallbacks):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")

    trace.get_current_span().set_attribute("function.name", function_name)
    trace.get_current_span().set_attribute("function.args", tool_call.function.arguments)

    function_call_result = await execute_tool_call(function_name, function_args, agent.tools, agent.mcp_servers)

    assert function_call_result is not None, f"Function {function_name} not implemented"
    trace.get_current_span().set_attribute("function.result", str(function_call_result))

    tool_return_summary = ""
    if isinstance(function_call_result, FinishTaskResult):
        agent.output = AgentOutput(
            result=function_call_result.result,
            summary=function_call_result.summary,
            feedback=function_call_result.feedback,
        )
        tool_return_summary = "Agent output set."
    elif isinstance(function_call_result, ShortenConversationResult):
        # Handle conversation shortening immediately
        start_message = create_start_message(agent)

        # Clear the conversation history
        agent.history = []

        # Add the start message
        append_user_message(
            agent.history,
            agent_callbacks,
            agent.name,
            start_message,
        )

        # Add the summary message
        append_user_message(
            agent.history,
            agent_callbacks,
            agent.name,
            f"A summary of your conversation with the client until now:\n\n{function_call_result.summary}\n\nPlease continue your work.",
        )

        tool_return_summary = "Conversation shortened and history reset."
    elif isinstance(function_call_result, TextResult):
        # Handle simple text results
        tool_return_summary = function_call_result.content
        if len(tool_return_summary) > 50_000:
            tool_return_summary = "System error: Tool call result too long."
    else:
        # This becomes a clear error case for an unsupported result type
        raise TypeError(f"Unknown tool result type: {type(function_call_result)}")

    append_tool_message(
        agent.history, agent_callbacks, agent.name, tool_call.id, function_name, function_args, tool_return_summary
    )


@tracer.start_as_current_span("do_single_step")
async def do_single_step(agent: Agent, agent_callbacks: AgentCallbacks, shorten_conversation_at_tokens: int):
    trace.get_current_span().set_attribute("agent.name", agent.name)

    if not any(tool.name() == "finish_task" for tool in agent.tools):
        raise RuntimeError("Agent needs to have a `finish_task` tool in order to run a step.")

    if not any(tool.name() == "shorten_conversation" for tool in agent.tools):
        raise RuntimeError("Agent needs to have a `shorten_conversation` tool in order to run a step.")

    tools = await get_tools(agent.tools, agent.mcp_servers)

    trace.get_current_span().set_attribute("agent.tools", json.dumps(tools))

    # Check that the agent has history
    if not agent.history:
        raise RuntimeError("Agent needs to have history in order to run a step.")

    trace.get_current_span().set_attribute("agent.history", json.dumps(agent.history))

    # Do one completion step
    completion = await complete(agent.history, model=agent.model, tools=tools)
    message = completion.message

    trace.get_current_span().set_attribute("completion.message", message.model_dump_json())

    # Remove the reasoning_content from the message, we cannot send it back to the LLM API.
    # At least DeepSeek complains about it.
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        trace.get_current_span().set_attribute("completion.reasoning_content", message.reasoning_content)
        del message.reasoning_content

    append_assistant_message(agent.history, agent_callbacks, agent.name, message)

    # Check if we need to do a tool call
    for tool_call in message.tool_calls or []:
        await handle_tool_call(tool_call, agent, agent_callbacks)

    if not message.tool_calls:
        append_user_message(
            agent.history,
            agent_callbacks,
            agent.name,
            "I detected a step from you without any tool calls. This is not allowed. If you want to ask the client something, please use the `ask_user` tool. If you are done with your task, please call the `finish_task` tool to signal that you are done. Otherwise, continue your work.",
        )

    # Note: We removed the token-based shortening prompt since conversation shortening
    # is now handled immediately when the shorten_conversation tool is called.
    # The agent can be instructed to call shorten_conversation through other means.

    return message


@tracer.start_as_current_span("run_agent_loop")
async def run_agent_loop(
    agent: Agent,
    agent_callbacks: AgentCallbacks,
    shorten_conversation_at_tokens=100_000,
) -> AgentOutput:
    if agent.output:
        raise RuntimeError("Agent already has a result or summary.")

    trace.get_current_span().set_attribute("agent.name", agent.name)

    parameters_json = json.dumps([dataclasses.asdict(p) for p in agent.parameters])
    trace.get_current_span().set_attribute("agent.parameter_description", parameters_json)

    start_message = create_start_message(agent)

    if agent.history:
        agent_callbacks.on_agent_start(agent.name, agent.model, is_resuming=True)
    else:
        agent_callbacks.on_agent_start(agent.name, agent.model, is_resuming=False)

    append_user_message(agent.history, agent_callbacks, agent.name, start_message)

    while True:
        while not agent.output:
            await do_single_step(agent, agent_callbacks, shorten_conversation_at_tokens)

        trace.get_current_span().set_attribute("agent.result", agent.output.result)
        trace.get_current_span().set_attribute("agent.summary", agent.output.summary)

        agent_callbacks.on_agent_end(agent.name, agent.output.result, agent.output.summary)

        if feedback := await agent.feedback_function(agent):
            formatted_feedback = FEEDBACK_TEMPLATE.format(
                feedback=textwrap.indent(feedback, "  "),
            )

            append_user_message(agent.history, agent_callbacks, agent.name, formatted_feedback)

            agent.output = None
        else:
            # Feedback was ok, so we can finish the agent.
            break

    if not agent.output:
        raise RuntimeError("Agent finished without a result.")

    return agent.output
