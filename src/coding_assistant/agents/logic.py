from dataclasses import dataclass
from dataclasses import field
import dataclasses
import functools
import json
import logging
import signal
import sys
import textwrap
import threading
from typing import Callable, Optional

from rich.prompt import Prompt
from rich import print
from rich.panel import Panel
from rich.pretty import Pretty
from opentelemetry import trace

from coding_assistant.config import Config
from coding_assistant.llm.model import complete
from coding_assistant.tools import Tool

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

PARAMETER_TEMPLATE = """
Name: {name}
Description: {description}
Value: {value}
""".strip()

SYSTEM_PROMPT_TEMPLATE = """
You are an agent named `{name}`.

## Task

Your client has been given the following description of your work and capabilities: 

{description}

## Parameters

Your client has provided the following parameters for your task:

{parameters}

## Instructions

The following are additional instructions provided by your client:

{instructions}
""".strip()

FEEDBACK_TEMPLATE = """
Your client has provided the following feedback on your work:

{feedback}

Please rework your result to address the feedback.
Afterwards, call the `finish_task` tool again to signal that you are done.
""".strip()


@dataclass
class Parameter:
    name: str
    description: str
    value: str


@dataclass
class AgentOutput:
    result: str
    summary: str


@dataclass
class Agent:
    name: str
    model: str

    description: str
    instructions: str
    parameters: list[Parameter]

    # This is a function that can validate an agents output.
    # If it returns a string, it will be given to the agent as feedback.
    feedback_function: Callable

    tools: list[Tool] = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)

    history: list = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    output: AgentOutput | None = None


class FinishTaskTool(Tool):
    def __init__(self, agent: Agent):
        self._agent = agent

    def name(self) -> str:
        return "finish_task"

    def description(self) -> str:
        return "Signals that the assigned task is complete. This tool must be called eventually to terminate the agent's execution loop. The final result and the summary of the work should be provided in the 'result' parameter, as this is the only output accessible to the client."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "The result of the work on the task. The work of the agent is evaluated based on this result.",
                },
                "summary": {
                    "type": "string",
                    "description": "A concise summary of the conversation the agent and the client had. There should be enough context such that the work could be continued based on this summary.",
                },
            },
            "required": ["result", "summary"],
        }

    async def execute(self, parameters) -> str:
        self._agent.output = AgentOutput(
            result=parameters["result"],
            summary=parameters["summary"],
        )
        return "Agent output set."


async def get_tools_from_mcp_servers(mcp_servers: list) -> list:
    tools = []
    for server in mcp_servers:
        for _, tool_list in await server.session.list_tools():
            for tool in tool_list or []:
                tool_id = f"mcp_{server.name}_{tool.name}"

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
        if name not in parameter_values:
            if name in required:
                raise RuntimeError(f"Parameter {name} is required but not provided.")
            else:
                continue

        # Convert all parameter values to sensible string representations
        parameter_type = parameter.get("type")
        if parameter_type == "string":
            value = parameter_values[name]

            if not isinstance(value, str):
                raise RuntimeError(f"Parameter {name} is not a string: {value}")
        elif parameter_type == "array":
            value = parameter_values[name]

            if not isinstance(value, list):
                raise RuntimeError(f"Parameter {name} is not an array: {value}")

            value = textwrap.indent("\n".join(value), "- ")
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
async def handle_tool_call(tool_call, agent: Agent):
    print(
        Panel(
            Pretty(tool_call.function),
            title=f"Agent {agent.name} tool call",
            border_style="green",
        ),
    )

    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments or "{}")

    trace.get_current_span().set_attribute("function.name", function_name)
    trace.get_current_span().set_attribute("function.args", tool_call.function.arguments)

    logger.debug(f"Calling tool {function_name} with args {function_args}")

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
    logger.debug(f"Function {function_name} returned {function_call_result}")

    if len(function_call_result) > 50_000:
        logger.warning(
            f"Function {function_name} returned too long result ({len(function_call_result)} characters). Trimming."
        )

        function_call_result = "System error: Tool call result too long. Please try again with different parameters."

    print(
        Panel(
            function_call_result,
            title=f"Tool {function_name} result",
            border_style="yellow",
        ),
    )

    agent.history.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_call_result,
        }
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


def create_system_message(agent: Agent) -> str:
    parameters_str = format_parameters(agent.parameters)
    return SYSTEM_PROMPT_TEMPLATE.format(
        name=agent.name,
        description=textwrap.indent(agent.description, "  "),
        parameters=textwrap.indent(parameters_str, "  "),
        instructions=textwrap.indent(agent.instructions, "  "),
    )


@tracer.start_as_current_span("do_single_step")
async def do_single_step(agent: Agent):
    trace.get_current_span().set_attribute("agent.name", agent.name)

    # Add the finish_task tool to the agent, if it is not already there.
    if not any(tool.name() == "finish_task" for tool in agent.tools):
        agent.tools.append(FinishTaskTool(agent))

    tools = []
    tools.extend(get_tools_from_agent(agent))
    tools.extend(await get_tools_from_mcp_servers(agent.mcp_servers))

    trace.get_current_span().set_attribute("agent.tools", json.dumps(tools))

    # If the agent has no history, this is our first step
    if not agent.history:
        system_message = create_system_message(agent)

        print(
            Panel(
                system_message,
                title=f"Agent {agent.name} starting",
                border_style="red",
            ),
        )

        agent.history.append(
            {
                "role": "system",
                "content": system_message,
            }
        )

    # Do one completion step
    trim_history(agent.history)

    trace.get_current_span().set_attribute("agent.history", json.dumps(agent.history))

    message = await complete(agent.history, model=agent.model, tools=tools)
    trace.get_current_span().set_attribute("completion.message", message.model_dump_json())

    # Remove the reasoning_content from the message, we cannot send it back to the LLM API.
    # At least DeepSeek complains about it.
    if hasattr(message, "reasoning_content"):
        trace.get_current_span().set_attribute("completion.reasoning_content", message.reasoning_content)
        del message.reasoning_content

    agent.history.append(message.model_dump())

    if message.content:
        print(
            Panel(
                message.content,
                title=f"Agent {agent.name} response",
                border_style="green",
            ),
        )

    # Check if we need to do a tool call
    for tool_call in message.tool_calls or []:
        await handle_tool_call(tool_call, agent)

    if not message.tool_calls:
        agent.history.append(
            {
                "role": "user",
                "content": "I detected a step from you without any tool calls. Remember to call the `finish_task` tool when you are done.",
            }
        )

    return message


@tracer.start_as_current_span("run_agent_loop")
async def run_agent_loop(
    agent: Agent,
) -> AgentOutput:
    if agent.output:
        raise RuntimeError("Agent already has a result or summary.")

    trace.get_current_span().set_attribute("agent.name", agent.name)

    parameters_json = json.dumps([dataclasses.asdict(p) for p in agent.parameters])
    trace.get_current_span().set_attribute("agent.parameter_description", parameters_json)

    while True:
        while not agent.output:
            await do_single_step(agent)

        trace.get_current_span().set_attribute("agent.result", agent.output.result)
        trace.get_current_span().set_attribute("agent.summary", agent.output.summary)

        print(
            Panel(
                f"Result: {agent.output.result}\n\nSummary: {agent.output.summary}",
                title=f"Agent {agent.name} result",
                border_style="red",
            ),
        )

        if feedback := await agent.feedback_function(agent):
            agent.feedback.append(feedback)

            feedback = FEEDBACK_TEMPLATE.format(
                feedback=textwrap.indent(feedback, "  "),
            )

            print(
                Panel(
                    feedback,
                    title=f"Agent {agent.name} feedback",
                    border_style="red",
                ),
            )

            agent.history.append(
                {
                    "role": "user",
                    "content": feedback,
                }
            )

            agent.output = None
        else:
            # Feedback was ok, so we can finish the agent.
            break

    if not agent.output:
        raise RuntimeError("Agent finished without a result.")

    return agent.output
