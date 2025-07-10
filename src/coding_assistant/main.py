import argparse
import asyncio
import logging
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    AsyncOpenAI,
    ItemHelpers,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    RunHooks,
    Runner,
    Tool,
)

from coding_assistant.agents.expert import create_expert_agent
from coding_assistant.agents.orchestrator import create_orchestrator_agent
from coding_assistant.agents.researcher import create_researcher_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools, get_filesystem_server, get_git_server


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--research", type=str, help="Question to ask the research agent."
    )
    parser.add_argument("--task", type=str, help="Task for the orchestrator agent.")
    parser.add_argument("--expert", type=str, help="Task for the expert agent.")
    parser.add_argument(
        "-w",
        "--working_directory",
        type=Path,
        help="The working directory to use.",
        default=Path(os.getcwd()),
    )
    return parser.parse_args()


logger = logging.getLogger(__name__)


def load_config(args) -> Config:
    model_name = os.environ.get("CODING_ASSISTANT_MODEL", "o4-mini")
    expert_model_name = os.environ.get("CODING_ASSISTANT_EXPERT_MODEL", "o1")

    model_factory = lambda: OpenAIChatCompletionsModel(model_name, AsyncOpenAI())
    expert_model_factory = lambda: OpenAIChatCompletionsModel(
        expert_model_name, AsyncOpenAI()
    )

    return Config(
        working_directory=args.working_directory,
        model_factory=model_factory,
        expert_model_factory=expert_model_factory,
    )


async def main():
    args = parse_args()
    config = load_config(args)

    os.chdir(config.working_directory)
    print(f"Running in working directory: {config.working_directory}")

    async with (
        get_filesystem_server(config) as filesystem_server,
        get_git_server(config) as git_server,
    ):
        mcp_servers = [filesystem_server, git_server]

        print("Available tools from MCP servers:")
        for server in mcp_servers:
            print(f"  - Server: {server.name}")
            for tool in await server.list_tools():
                print(f"    - {tool.name}: {tool.description}")

        tools = Tools(mcp_servers=mcp_servers)

        if args.research:
            agent_to_run = create_researcher_agent(config, tools)
            initial_input = args.research
        elif args.task:
            agent_to_run = create_orchestrator_agent(config, tools)
            initial_input = args.task
        elif args.expert:
            agent_to_run = create_expert_agent(config, tools)
            initial_input = args.expert
        else:
            print("No task or question specified.")
            sys.exit(1)

        result = Runner.run_streamed(agent_to_run, initial_input)

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                continue

            if event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
                continue

            if event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print("-- Tool was called")
                elif event.item.type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
                elif event.item.type == "message_output_item":
                    print(
                        f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                    )

        print(result.final_output_as(str))


if __name__ == "__main__":
    asyncio.run(main())
