import logging
from typing import Annotated, List

from rich.console import Console
from smolagents import CodeAgent, MultiStepAgent, Tool, tool

from coding_assistant.config import Config
from coding_assistant.tools.tools import get_file_tools

console = Console()
logger = logging.getLogger(__name__)

RESEARCHER_DESCRIPTION = """
Researcher agent, which is responsible for answering questions.
These can be general questions or questions about the code base.
This agent cannot implement changes to the code base.
""".strip()


def create_researcher_agent(config: Config) -> MultiStepAgent:
    tools = []
    tools.extend(get_file_tools(config))

    return CodeAgent(
        model=config.model_factory(),
        tools=tools,
        name="Researcher",
        description=RESEARCHER_DESCRIPTION,
    )
