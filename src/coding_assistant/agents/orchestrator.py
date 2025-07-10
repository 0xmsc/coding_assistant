import logging
from typing import Annotated

from coding_assistant.agents.agent import Agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

ORCHESTRATOR_INSTRUCTIONS = """
You are an Orchestrator agent. Your goal is to coordinate other specialized agents to efficiently complete complex tasks.
""".strip()


def create_orchestrator_agent(task: str, config: Config, tools: Tools) -> Agent:
    return Agent(
        name="orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        mcp_servers=tools.mcp_servers,
        tools=[],
        model=config.model,
        task=task,
        history=[],
    )
