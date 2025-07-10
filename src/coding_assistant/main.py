import argparse
import logging
import os
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

from smolagents import OpenAIServerModel

from coding_assistant.agents.orchestrator import create_orchestrator_agent
from coding_assistant.config import Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--research", type=str, help="Question to ask the research agent."
    )
    parser.add_argument("--task", type=str, help="Question to ask the research agent.")
    parser.add_argument("--expert", type=str, help="Question to ask the expert agent.")
    parser.add_argument(
        "-w",
        "--working_directory",
        type=Path,
        help="The working directory to use.",
        default=Path(os.getcwd()),
    )
    parser.add_argument(
        "--user-feedback",
        default=True,
        action=BooleanOptionalAction,
        help="Whether to ask for user feedback.",
    )
    parser.add_argument(
        "--default-model",
        default="o3-mini",
        type=str,
        help="Default model to use.",
    )
    parser.add_argument(
        "--expert-model",
        default="o1",
        type=str,
        help="Expert model to use.",
    )
    return parser.parse_args()


logger = logging.getLogger(__name__)


def create_config(args: argparse.Namespace) -> Config:
    return Config(
        working_directory=args.working_directory,
        model_factory=lambda: OpenAIServerModel(model_id=args.default_model),
        expert_model_factory=lambda: OpenAIServerModel(model_id=args.expert_model),
    )


def main():
    logging.basicConfig(
        format="%(name)s:%(filename)s:%(lineno)d:%(funcName)s:%(levelname)s: %(message)s"
    )
    logging.getLogger("coding_assistant").setLevel(logging.DEBUG)

    args = parse_args()
    config = create_config(args)

    if args.research:
        pass
    elif args.task:
        create_orchestrator_agent(config).run(args.task)
    elif args.expert:
        pass
    else:
        print("No task specified.")
