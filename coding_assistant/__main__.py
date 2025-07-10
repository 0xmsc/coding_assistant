import logging
import os
from pathlib import Path
from argparse import ArgumentParser

from coding_assistant.agents.orchestrator import run_orchestrator_agent
from coding_assistant.agents.researcher import run_research_agent
from coding_assistant.config import get_global_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--research", type=str, help="Question to ask the research agent.")
    parser.add_argument("--task", type=str, help="Question to ask the research agent.")
    return parser.parse_args()


def main():
    logging.basicConfig(format="%(name)s:%(filename)s:%(lineno)d:%(funcName)s:%(levelname)s: %(message)s")
    logging.getLogger("coding_assistant").setLevel(logging.DEBUG)

    working_directoy: Path = Path(os.getcwd())
    logging.debug(f"Working directory: {working_directoy}")

    get_global_config().working_directory = working_directoy

    args = parse_args()

    if args.research:
        run_research_agent(
            question=args.research,
        )
    if args.task:
        run_orchestrator_agent(
            task=args.task,
        )
    else:
        print("No task specified.")


if __name__ == "__main__":
    main()
