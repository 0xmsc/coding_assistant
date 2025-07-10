import logging
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

RESEARCHER_INSTRUCTIONS = """
You are a Researcher agent. Your responsibility is to answer questions accurately.
These questions can be general knowledge questions or specific questions about the provided code base.
You cannot implement changes to the code base.
You have access to tools for filesystem operations, web searching, etc. Use them to find the necessary information.
""".strip()


def create_researcher_agent(config: Config, tools: Tools) -> Agent:
    return Agent(
        name="researcher",
        instructions=RESEARCHER_INSTRUCTIONS,
        mcp_servers=tools.mcp_servers,
        model=config.model_factory(),
    )
