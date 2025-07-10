import logging
import os
from pathlib import Path

from agents.research import run_research_agent


def main():
    logging.basicConfig(
        level=logging.DEBUG, format="$(filename):$(lineno):$(levelname)s: $(message)s"
    )

    working_directoy: Path = Path(os.getcwd())
    logging.debug(f"Working directory: {working_directoy}")

    run_research_agent(
        task="What agents are there in the project?", working_directory=working_directoy
    )


if __name__ == "__main__":
    main()
