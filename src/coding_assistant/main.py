import argparse
import asyncio
import logging
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Runner

from coding_assistant.agents.expert import create_expert_agent
from coding_assistant.agents.orchestrator import create_orchestrator_agent
from coding_assistant.agents.researcher import create_researcher_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools, get_filesystem_server


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

    async with get_filesystem_server(config) as filesystem_server:
        tools = Tools(mcp_servers=[filesystem_server])

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

        result = await Runner.run(agent_to_run, initial_input)
        print(result.final_output_as(str))


if __name__ == "__main__":
    asyncio.run(main())
