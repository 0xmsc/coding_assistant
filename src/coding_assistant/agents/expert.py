import logging
from typing import Annotated, List

from smolagents import CodeAgent, MultiStepAgent, Tool, tool

from coding_assistant.config import Config, get_global_config
from coding_assistant.tools.file import read_only_file_tools
from coding_assistant.tools.notebook import get_notebook_tools

logger = logging.getLogger(__name__)

EXPERT_DESCRIPTION = """
Expert agent, which is responsible for dealing with exceptional tasks or queries.
The agent is expected to have expert level knowledge in software engineering and related fields.
""".strip()


def create_expert_tools() -> List[Tool]:
    tools = []
    tools.extend(read_only_file_tools())
    tools.extend(get_notebook_tools())
    return tools


def create_expert_agent(config: Config) -> MultiStepAgent:
    return CodeAgent(
        model=config.expert_model_factory(),
        tools=create_expert_tools(),
        name="Expert",
        description=EXPERT_DESCRIPTION,
    )
