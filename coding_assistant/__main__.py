import logging
import os
from pathlib import Path
from argparse import ArgumentParser

from coding_assistant.agents.orchestrator import run_orchestrator_agent
from coding_assistant.agents.researcher import run_researcher_agent
from coding_assistant.config import get_global_config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--research", type=str, help="Question to ask the research agent.")
    parser.add_argument("--task", type=str, help="Question to ask the research agent.")
    parser.add_argument(
        "--working_directory", type=Path, help="The working directory to use.", default=Path(os.getcwd())
    )
    return parser.parse_args()


logger = logging.getLogger(__name__)


def main():
    args = parse_args()

    logging.basicConfig(format="%(name)s:%(filename)s:%(lineno)d:%(funcName)s:%(levelname)s: %(message)s")
    logging.getLogger("coding_assistant").setLevel(logging.DEBUG)

    working_directoy = args.working_directory
    os.chdir(working_directoy)
    print(f"Running in working directory: {working_directoy}")

    get_global_config().working_directory = working_directoy

    if os.environ.get("BMW_API_KEY"):
        from bmw_llm_adapter.langchain import BMWModel
        from bmw_llm_adapter.bmw_api_model import ModelName
        get_global_config().model_factory = lambda: BMWModel(model_name=ModelName.gpt_4o)

    assert get_global_config().model_factory, "No model factory set."

    if args.research:
        run_researcher_agent(
            question=args.research,
            ask_user_for_feedback=True,
        )
    elif args.task:
        run_orchestrator_agent(
            task=args.task,
        )
    else:
        print("No task specified.")


if __name__ == "__main__":
    main()
