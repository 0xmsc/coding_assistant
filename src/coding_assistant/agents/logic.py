import dataclasses
import functools
import json
import logging
import signal
import sys
import textwrap
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

from opentelemetry import trace
from rich import print

from coding_assistant.agents.callbacks import AgentCallbacks
from coding_assistant.config import Config
from coding_assistant.llm.model import complete

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

PARAMETER_TEMPLATE = """
Name: {name}
Description: {description}
Value: {value}
""".strip()

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


class Tool(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def parameters(self) -> dict: ...

    @abstractmethod
    async def execute(self, parameters) -> str: ...


@dataclass
class Parameter:
    name: str
    description: str
    value: str


@dataclass
class AgentOutput:
    result: str
    summary: str
    feedback: str | None


@dataclass
class Agent:
    name: str
    model: str

    description: str

    parameters: list[Parameter]

    # This is a function that can validate an agents output.
    # If it returns a string, it will be given to the agent as feedback.
    feedback_function: Callable

    tools: list[Tool] = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)

    history: list = field(default_factory=list)
    shortened_conversation: str | None = None

    output: AgentOutput | None = None


def append_tool_message(
    history: list,
    callbacks: AgentCallbacks,
    agent_name: str,
    tool_call_id: str,
    function_name: str,
    function_args: dict,
    function_call_result: str,
):
    callbacks.on_tool_message(agent_name, function_name, function_args, function_call_result)

    history.append(
        {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": function_name,
            "content": function_call_result,
        }
    )


def append_user_message(history: list, callbacks: AgentCallbacks, agent_name: str, content: str):
    callbacks.on_user_message(agent_name, content)

    history.append(
        {
            "role": "user",
            "content": content,
        }
    )


def append_assistant_message(history: list, callbacks: AgentCallbacks, agent_name: str, message):
    if message.content:
        callbacks.on_assistant_message(agent_name, message.content)

    history.append(message.model_dump())


def fix_input_schema(input_schema: dict):
    """
    Fixes the input schema to be compatible with Gemini API
    This is a workaround for the fact that Gemini API does not support certain values for the `format` field
    """

    for property in input_schema.get("properties", {}).values():
        if (format := property.get("format")) and format == "uri":
            # Gemini API does not support `format: uri`, so we remove it
            property.pop("format", None)


async def get_tools_from_mcp_servers(mcp_servers: list) -> list:
    tools = []
    for server in mcp_servers:
        for _, tool_list in await server.session.list_tools():
            for tool in tool_list or []:
                tool_id = f"mcp_{server.name}_{tool.name}"

                fix_input_schema(tool.inputSchema)

                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_id,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
    return tools


def fill_parameters(
    parameter_description: dict,
    parameter_values: dict,
) -> list[Parameter]:
    parameters = []

    required = set(parameter_description.get("required", []))

    for name, parameter in parameter_description["properties"].items():
        # Check if required parameters are provided
        if name not in parameter_values or parameter_values[name] is None:
            if name in required:
                raise RuntimeError(f"Parameter {name} is required but not provided.")
            else:
                continue

        # Convert all parameter values to sensible string representations
        parameter_type = parameter.get("type")
        if parameter_type in "string":
            if not isinstance(parameter_values[name], str):
                raise RuntimeError(f"Parameter {name} is not a string: {value}")
            value = parameter_values[name]
        elif parameter_type == "array":
            if not isinstance(parameter_values[name], list):
                raise RuntimeError(f"Parameter {name} is not an array: {value}")
            value = textwrap.indent("\n".join(parameter_values[name]), "- ")
        elif parameter_type == "boolean":
            if not isinstance(parameter_values[name], bool):
                raise RuntimeError(f"Parameter {name} is not a boolean: {value}")
            value = str(parameter_values[name])
        else:
            raise RuntimeError(f"Unsupported parameter type: {parameter_type} for parameter {name}")

        parameters.append(
            Parameter(
                name=name,
                description=parameter["description"],
                value=value,
            )
        )

    return parameters


def get_tools_from_agent(agent: Agent) -> list:
    result = []
    for tool in agent.tools:
        if tool.name().startswith("mcp_"):
            raise RuntimeError("Tools cannot start with mcp_")

        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name(),
                    "description": tool.description(),
                    "parameters": tool.parameters(),
                },
            }
        )
    return result


async def handle_mcp_tool_call(function_name, arguments, mcp_servers):
    parts = function_name.split("_")
    assert parts[0] == "mcp"

    server_name = parts[1]
    tool_name = "_".join(parts[2:])

    for server in mcp_servers:
        if server.name == server_name:
            result = await server.session.call_tool(tool_name, arguments)
            if not result.content:
                return "MCP server did not return any content."
            return result.content[0].text

    raise RuntimeError(f"Server {server_name} not found in MCP servers.")


@tracer.start_as_current_span("handle_tool_call")
async def handle_tool_call(tool_call, agent: Agent, agent_callbacks: AgentCallbacks):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")

    trace.get_current_span().set_attribute("function.name", function_name)
    trace.get_current_span().set_attribute("function.args", tool_call.function.arguments)

    function_call_result = None

    if function_name.startswith("mcp_"):
        function_call_result = await handle_mcp_tool_call(function_name, function_args, agent.mcp_servers)
    else:
        for tool in agent.tools:
            if tool.name() == function_name:
                function_call_result = await tool.execute(function_args)
                break
        else:
            raise RuntimeError(f"Tool {function_name} not found in agent tools.")

    assert function_call_result is not None, f"Function {function_name} not implemented"
    trace.get_current_span().set_attribute("function.result", function_call_result)

    if len(function_call_result) > 50_000:
        function_call_result = "System error: Tool call result too long. Please try again with different parameters."

    append_tool_message(
        agent.history, agent_callbacks, agent.name, tool_call.id, function_name, function_args, function_call_result
    )


def trim_history(history: list):
    pass


def format_parameters(parameters: list[Parameter]) -> str:
    parameter_descriptions = []

    for parameter in parameters:
        value_str = parameter.value

        if "\n" in value_str:
            value_str = "\n" + textwrap.indent(value_str, "  ")

        parameter_descriptions.append(
            PARAMETER_TEMPLATE.format(
                name=parameter.name,
                description=parameter.description,
                value=value_str,
            )
        )

    return "\n\n".join(parameter_descriptions)


def create_start_message(agent: Agent) -> str:
    parameters_str = format_parameters(agent.parameters)
    return START_MESSAGE_TEMPLATE.format(
        name=agent.name,
        description=textwrap.indent(agent.description, "  "),
        parameters=textwrap.indent(parameters_str, "  "),
    )


@tracer.start_as_current_span("do_single_step")
async def do_single_step(agent: Agent, agent_callbacks: AgentCallbacks, shorten_conversation_at_tokens: int):
    trace.get_current_span().set_attribute("agent.name", agent.name)

    if not any(tool.name() == "finish_task" for tool in agent.tools):
        raise RuntimeError("Agent needs to have a `finish_task` tool in order to run a step.")

    tools = []
    tools.extend(get_tools_from_agent(agent))
    tools.extend(await get_tools_from_mcp_servers(agent.mcp_servers))

    trace.get_current_span().set_attribute("agent.tools", json.dumps(tools))

    # Check that the agent has history
    if not agent.history:
        raise RuntimeError("Agent needs to have history in order to run a step.")

    # Trim the history if its getting too big
    trim_history(agent.history)

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
            "I detected a step from you without any tool calls. This is not allowed. If you want to ask the client something, please use the `ask_user` tool. Otherwise, please call the `finish_task` tool to signal that you are done.",
        )
    elif completion.tokens > shorten_conversation_at_tokens and not agent.shortened_conversation:
        append_user_message(
            agent.history,
            agent_callbacks,
            agent.name,
            "Your conversation is becoming too long. Please call `shorten_conversation` to trim it.",
        )

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

            if agent.shortened_conversation:
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
                    f"A summary of your conversation with the client until now:\n\n{agent.shortened_conversation}\n\nPlease continue your work.",
                )

                agent.shortened_conversation = None

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
