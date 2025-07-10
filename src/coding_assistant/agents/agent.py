import copy
from dataclasses import dataclass
from dataclasses import field
import json
import logging
from typing import Optional

from coding_assistant.llm.model import complete

logger = logging.getLogger(__name__)


def get_default_functions() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "finish_task",
                "description": "Finish the task. Only call this when the task is done.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "result": {
                            "type": "string",
                            "description": "The result of the task, if any.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the user for input.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the user.",
                        },
                    },
                    "required": ["question"],
                },
            },
        },
    ]


@dataclass
class Agent:
    name: str
    model: str
    instructions: str
    task: str

    tools: list = field(default_factory=list)
    mcp_servers: list = field(default_factory=list)

    history: list = field(default_factory=list)
    finished: bool = False
    result: Optional[str] = None


def handle_tool_call(tool_call, agent: Agent):
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)

    logger.debug(f"Calling tool {function_name} with args {function_args}")

    if function_name == "finish_task":
        agent.finished = True
        agent.result = function_args.get("result", "")
        return

    function_call_result = None

    if function_name == "ask_user":
        question = function_args.get("question")
        function_call_result = input(f"Agent {agent.name} asks: {question}\nAnswer: ")

    assert function_call_result is not None, f"Function {function_name} not implemented"

    agent.history.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_call_result,
        }
    )


def do_single_step(agent: Agent):
    tools = list(agent.tools)
    tools.extend(get_default_functions())

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
    completion = complete(agent.history, model=agent.model, tools=tools)
    logger.info(f"Got completion {completion} from LLM")

    message = completion["choices"][0]["message"]
    agent.history.append(dict(message))

    # Check if we need to do a tool call
    for tool_call in message.tool_calls or []:
        handle_tool_call(tool_call, agent)

    return message


def run_agent_loop(agent: Agent):
    step_counter = 0
    while not agent.finished:
        step = do_single_step(agent)
        if step.content:
            logger.info(f"[{step_counter}] Agent {agent.name}: {step.content}")
        step_counter += 1
    return agent.result
