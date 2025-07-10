import argparse
import logging
import os
from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatLiteLLM

from coding_assistant.agents.expert import run_expert_agent
from coding_assistant.agents.orchestrator import run_orchestrator_agent
from coding_assistant.agents.researcher import run_researcher_agent
from coding_assistant.config import get_global_config, Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--research", type=str, help="Question to ask the research agent.")
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
        "--user-feedback", default=True, action=BooleanOptionalAction, help="Whether to ask for user feedback."
    )
    return parser.parse_args()


logger = logging.getLogger(__name__)


def load_config(args: argparse.Namespace) -> Config:
    backend = os.environ.get("CODING_ASSISTANT_BACKEND", "OPENAI").upper()
    model = os.environ.get("CODING_ASSISTANT_MODEL", "o3-mini")
    reasoning_model = os.environ.get("CODING_ASSISTANT_REASONING_MODEL", "o3-mini")

    if backend == "OPENAI":
        model_factory = lambda: ChatOpenAI(model=model)
        reasoning_model_factory = lambda: ChatOpenAI(model=reasoning_model)

    elif backend == "OPENROUTER":
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        custom_llm_provider = "openrouter"
        model_factory = lambda: ChatLiteLLM(
            model=model, base_url=base_url, api_key=api_key, custom_llm_provider=custom_llm_provider
        )
        reasoning_model_factory = lambda: ChatLiteLLM(
            model=reasoning_model, base_url=base_url, api_key=api_key, custom_llm_provider=custom_llm_provider
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")

    config = get_global_config()
    config.model_factory = model_factory
    config.reasoning_model_factory = reasoning_model_factory
    config.working_directory = args.working_directory
    return config


def main():
    args = parse_args()

    logging.basicConfig(format="%(name)s:%(filename)s:%(lineno)d:%(funcName)s:%(levelname)s: %(message)s")
    logging.getLogger("coding_assistant").setLevel(logging.DEBUG)

    config = load_config(args)

    os.chdir(config.working_directory)
    print(f"Running in working directory: {config.working_directory}")

    if args.research:
        run_researcher_agent(
            question=args.research,
            ask_user_for_feedback=args.user_feedback,
            notebook=dict(),
        )
    elif args.task:
        run_orchestrator_agent(
            task=args.task,
            ask_user_for_feedback=args.user_feedback,
            notebook=dict(),
        )
    elif args.expert:
        run_expert_agent(
            task=args.expert,
            ask_user_for_feedback=args.user_feedback,
            notebook=dict(),
        )
    else:
        print("No task specified.")


if __name__ == "__main__":
    main()
