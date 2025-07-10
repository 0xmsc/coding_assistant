import logging
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

EXPERT_INSTRUCTIONS = """
You are an Expert agent. Handle exceptional tasks or queries requiring deep expertise.
Leverage your expert-level knowledge in software engineering and related fields to provide insightful analysis, solutions, or advice.
Use the provided tools to gather necessary information from the file system or notebook context.
""".strip()


def create_expert_agent(config: Config, tools: Tools) -> Agent:
    return Agent(
        name="expert",
        instructions=EXPERT_INSTRUCTIONS,
        mcp_servers=tools.mcp_servers,
        model=config.expert_model_factory(),
    )
