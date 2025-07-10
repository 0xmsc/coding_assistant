import logging
from typing import Annotated, List

from smolagents import CodeAgent, MultiStepAgent, Tool

from coding_assistant.agents.researcher import create_researcher_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

DEVELOPER_DESCRIPTION = """
Developer agent, which is responsible for carrying out implementation plans.
The agent is not responsible for planning implementations or making decisions about software architecture.
The agent should receive detailed instructions on how to implement a task, and execute those instructions.
""".strip()


def create_developer_agent(config: Config, tools: Tools) -> MultiStepAgent:
    return CodeAgent(
        model=config.model_factory(),
        tools=[*tools.file_tools],
        managed_agents=[
            create_researcher_agent(config, tools),
        ],
        name="Developer",
        description=DEVELOPER_DESCRIPTION,
    )
