import logging
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.agents.developer import create_developer_tool
from coding_assistant.agents.expert import create_expert_tool
from coding_assistant.agents.planner import create_planner_tool
from coding_assistant.agents.researcher import create_researcher_tool
from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

ORCHESTRATOR_INSTRUCTIONS = """
You are an Orchestrator agent. Your goal is to coordinate other specialized agents to efficiently complete complex tasks.
""".strip()


def create_orchestrator_agent(config: Config, tools: Tools) -> Agent:
    return Agent(
        name="orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        mcp_servers=tools.mcp_servers,
        tools=[
            create_planner_tool(config, tools),
            create_researcher_tool(config, tools),
            create_developer_tool(config, tools),
            create_expert_tool(config, tools),
        ],
        model=config.expert_model_factory(),
    )
