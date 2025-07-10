import copy
from dataclasses import dataclass
from dataclasses import field
import json
import logging
from typing import Optional

from rich import print
from rich.panel import Panel
from rich.pretty import Pretty
from opentelemetry import trace

from coding_assistant.llm.model import complete
from coding_assistant.tools import Tool

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


@dataclass
class Agent:
    name: str
    model: str
    instructions: str
    task: str

    tools: list[Tool] = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)

    history: list = field(default_factory=list)
    finished: bool = False
    result: Optional[str] = None


class FinishTaskTool(Tool):
    def name(self) -> str:
        return "finish_task"

    def description(self) -> str:
        return "Finish the task. Only call this when the task is done. Note that this function has to bed called at some point in time, otherwise the agent will be in an infinite loop."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "The result of the task, if any.",
                },
            },
        }

    def execute(self, parameters) -> str:
        assert False, "FinishTaskTool should not be executed directly."


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
        agent.finished = True
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
    for entry in history:
        if entry["role"] == "tool" and len(entry["content"]) > 20_000:
            logger.warning(
                f"Tool call content too long ({len(entry['content'])} characters)."
            )

            entry["content"] = (
                "!!! This tool output has been removed because it was too long, try other parameters"
            )


@tracer.start_as_current_span("do_single_step")
async def do_single_step(agent: Agent):
    trace.get_current_span().set_attribute("agent.name", agent.name)
    trace.get_current_span().set_attribute("agent.task", agent.task)
    trace.get_current_span().set_attribute("agent.history", json.dumps(agent.history))

    tools = []
    tools.extend(get_tools_from_agent(agent))
    tools.extend(await get_tools_from_mcp_servers(agent.mcp_servers))

    # If the agent has no history, this is our first step
    if not agent.history:
        agent.history.append(
            {
                "role": "system",
                "content": agent.instructions,
            }
        )

        agent.history.append(
            {
                "role": "user",
                "content": agent.task,
            }
        )

    # Do one completion step
    trim_history(agent.history)
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


@tracer.start_as_current_span("run_agent_loop")
async def run_agent_loop(agent: Agent):
    trace.get_current_span().set_attribute("agent.name", agent.name)
    trace.get_current_span().set_attribute("agent.task", agent.task)

    step_counter = 0
    while not agent.finished:
        await do_single_step(agent)
        step_counter += 1

    assert agent.result
    trace.get_current_span().set_attribute("agent.result", agent.result)
    return agent.result
