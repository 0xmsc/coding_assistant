import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.agents.expert import create_expert_tool
from coding_assistant.agents.researcher import create_researcher_tool
from coding_assistant.config import Config
from coding_assistant.tools import Tools

PLANNER_INSTRUCTIONS = f"""
You are a Planner agent. 
Your primary responsibility is to create a detailed implementation plan for a given task.
""".strip()

logger = logging.getLogger(__name__)


def create_planner_agent(config: Config, tools: Tools) -> Agent:
    return Agent(
        name="planner",
        instructions=PLANNER_INSTRUCTIONS,
        tools=[
            create_researcher_tool(config, tools),
            create_expert_tool(config, tools),
        ],
        mcp_servers=tools.mcp_servers,
        model=config.model_factory(),
    )


def create_planner_tool(config, tools: Tools) -> Tool:
    return create_planner_agent(config, tools).as_tool(
        "planner_tool",
        tool_description="Create a detailed implementation plan for a task.",
    )
