import logging
from typing import Annotated, List

from smolagents import CodeAgent, MultiStepAgent, Tool, ToolCallingAgent

from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

RESEARCHER_DESCRIPTION = """
Researcher agent, which is responsible for answering questions.
These can be general questions or questions about the code base.
This agent cannot implement changes to the code base.
This agent can access the filesystem, the web, etc.
""".strip()


def create_researcher_agent(config: Config, tools: Tools) -> MultiStepAgent:
    return CodeAgent(
        model=config.model_factory(),
        tools=[*tools.file_tools],
        name="researcher",
        description=RESEARCHER_DESCRIPTION,
    )
