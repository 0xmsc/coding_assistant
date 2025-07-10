import argparse
import asyncio
import logging
import os
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import Any

from coding_assistant.agents.agent import do_single_step, run_agent_loop
from coding_assistant.agents.orchestrator import create_orchestrator_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools, get_filesystem_server


logging.basicConfig(level=logging.WARNING)

logger = logging.getLogger("coding_assistant")
logger.setLevel(logging.DEBUG)


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


def load_config(args) -> Config:
    model_name = os.environ.get("CODING_ASSISTANT_MODEL", "o4-mini")
    expert_model_name = os.environ.get("CODING_ASSISTANT_EXPERT_MODEL", "<disabled>")

    return Config(
        working_directory=args.working_directory,
        model=model_name,
        expert_model=expert_model_name,
    )


async def _main():
    args = parse_args()
    config = load_config(args)

    os.chdir(config.working_directory)
    logger.info(f"Running in working directory: {config.working_directory}")

    task = "Update the contents of README.md, if necessary."

    async with get_filesystem_server(config) as filesystem_server:
        tools = Tools(mcp_servers=[filesystem_server])

        if task:
            agent = create_orchestrator_agent(task, config, tools)
        elif args.task:
            agent = create_orchestrator_agent(args.task, config, tools)
        else:
            print("No task or question specified.")
            sys.exit(1)

        await run_agent_loop(agent)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
