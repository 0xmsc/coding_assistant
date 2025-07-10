from dataclasses import dataclass
from dataclasses import field
import dataclasses
import json
import logging
import textwrap
from typing import Optional

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

## Result

It is crucial that you return all results in the result parameter of the finish_task tool call.
Your client does not have access to any other output from you.
""".strip()


@dataclass
class Parameter:
    name: str
    description: str
    value: str


@dataclass
class Agent:
    name: str
    model: str

    description: str
    parameters: list[Parameter]

    tools: list[Tool] = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)

    history: list = field(default_factory=list)
    result: str | None = None


class FeedbackTool(Tool):
    def __init__(self, config: Config):
        self._config = config

    def name(self) -> str:
        return "launch_feedback_agent"

    def description(self) -> str:
        return "Launch a feedback agent that provides feedback on the output of another agent. This agent evaluates whether the output is acceptable for a given task. If it is, the feedback agent will finish its task with only the output 'Ok' and nothing else. If it is not, the feedback agent will output what is wrong with the output and how it needs to be improved."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task that was given to the agent.",
                },
                "output": {
                    "type": "string",
                    "description": "The output of the agent.",
                },
            },
            "required": ["task", "output"],
        }

    async def execute(self, parameters: dict) -> str:
        feedback_agent = Agent(
            name="Feedback",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=[],
            tools=[],
            model=self._config.model,
        )

        return await run_agent_loop(
            feedback_agent,
            self._config,
            ask_user_for_feedback=False,
            # We cannot ask the feedback agent for feedback, as we will end up in an infinite loop.
            ask_agent_for_feedback=False,
        )


class FinishTaskTool(Tool):
    def name(self) -> str:
        return "finish_task"

    def description(self) -> str:
        return "Signals that the assigned task is complete. This tool must be called eventually to terminate the agent's execution loop. The final result or summary of the task should be provided in the 'result' parameter, as this is the only output accessible to the client. Your client should understand what you've done while working on the task by only looking at the `result` parameter."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "The result of the task or a summary of the work that has been done.",
                },
            },
            "required": ["result"],
        }

    async def execute(self, _) -> str:
        raise RuntimeError("FinishTaskTool should not be executed directly.")


def get_default_functions() -> list:
    return [FinishTaskTool()]


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
        value = parameter_values.get(name)

        if not value:
            if name in required:
                raise RuntimeError(f"Parameter {name} is required but not provided.")
            else:
                continue

        parameters.append(
            Parameter(
                name=name,
                description=parameter["description"],
                value=value,
            )
        )

    return parameters


def get_tools_from_agent(agent: Agent) -> list:
    tools = []
    tools.extend(get_default_functions())
    tools.extend(agent.tools)

    result = []
    for tool in tools:
        assert not tool.name().startswith("mcp_")
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
    function_args = json.loads(tool_call.function.arguments)

    trace.get_current_span().set_attribute("function.name", function_name)
    trace.get_current_span().set_attribute(
        "function.args", tool_call.function.arguments
    )

    logger.debug(f"Calling tool {function_name} with args {function_args}")

    if function_name == "finish_task":
        agent.result = function_args.get("result", "")
        return

    function_call_result = None

    if function_name.startswith("mcp_"):
        function_call_result = await handle_mcp_tool_call(
            function_name, function_args, agent.mcp_servers
        )
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

    if len(function_call_result) > 20_000:
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


def create_system_message(agent: Agent) -> str:
    parameter_descriptions = []

    for parameter in agent.parameters:
        parameter_descriptions.append(
            PARAMETER_TEMPLATE.format(**dataclasses.asdict(parameter))
        )

    parameters_str = "\n\n".join(parameter_descriptions)

    return SYSTEM_PROMPT_TEMPLATE.format(
        name=agent.name,
        description=textwrap.indent(agent.description, "  "),
        parameters=textwrap.indent(parameters_str, "  "),
    )


@tracer.start_as_current_span("do_single_step")
async def do_single_step(agent: Agent):
    trace.get_current_span().set_attribute("agent.name", agent.name)

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

    completion = complete(agent.history, model=agent.model, tools=tools)
    logger.debug(f"Got completion {completion} from LLM")

    message = completion["choices"][0]["message"]
    trace.get_current_span().set_attribute(
        "completion.message", message.model_dump_json()
    )

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

    return message


async def get_feedback(
    agent: Agent,
    config: Config,
    ask_user_for_feedback: bool,
    ask_agent_for_feedback: bool,
) -> str | None:
    feedback = None

    if ask_user_for_feedback:
        feedback = Prompt.ask("Feedback:", default="Ok")
    elif ask_agent_for_feedback:
        # Spawn a feedback agent
        feedback_tool = FeedbackTool(config)
        feedback = await feedback_tool.execute(
            parameters={
                # Give the system message as the task.
                "task": agent.history[0]["content"],
                "output": agent.result,
            }
        )

    if not feedback:
        return None
    elif feedback == "Ok":
        return None
    else:
        return feedback


@tracer.start_as_current_span("run_agent_loop")
async def run_agent_loop(
    agent: Agent,
    config: Config,
    ask_user_for_feedback: bool = False,
    ask_agent_for_feedback: bool = True,
) -> str:
    trace.get_current_span().set_attribute("agent.name", agent.name)

    parameters_json = json.dumps([dataclasses.asdict(p) for p in agent.parameters])
    trace.get_current_span().set_attribute(
        "agent.parameter_description", parameters_json
    )

    while True:
        # Run the agent until it finishes
        while not agent.result:
            await do_single_step(agent)

        trace.get_current_span().set_attribute("agent.result", agent.result)

        print(
            Panel(
                agent.result,
                title=f"Agent {agent.name} result",
                border_style="red",
            ),
        )

        if feedback := await get_feedback(
            agent,
            config,
            ask_user_for_feedback=ask_user_for_feedback,
            ask_agent_for_feedback=ask_agent_for_feedback,
        ):
            print(
                Panel(
                    feedback,
                    title=f"Agent {agent.name} feedback",
                    border_style="red",
                ),
            )

            # Remove the finish_task tool call from the history
            agent.history.pop()
            agent.result = None

            agent.history.append(
                {
                    "role": "user",
                    "content": feedback,
                }
            )
        else:
            # Feedback was ok, so we can finish the agent.
            break

    return agent.result
