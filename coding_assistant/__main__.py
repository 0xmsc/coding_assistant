import logging
import os
from pathlib import Path

from coding_assistant.agents.research import run_research_agent


def main():
    logging.basicConfig(format="%(name)s:%(filename)s:%(lineno)d:%(funcName)s:%(levelname)s: %(message)s")
    logging.getLogger("coding_assistant").setLevel(logging.DEBUG)

    working_directoy: Path = Path(os.getcwd())
    logging.debug(f"Working directory: {working_directoy}")

    run_research_agent(
        question="What agents are there in the project? And what are their main tasks? What tools do they use?",
        working_directory=working_directoy,
    )


if __name__ == "__main__":
    main()
