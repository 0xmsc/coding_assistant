import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.agents.expert import create_expert_agent
from coding_assistant.agents.researcher import create_researcher_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools

PLANNER_INSTRUCTIONS = f"""
You are a Planner agent. Your primary responsibility is to create a detailed implementation plan for a given task.
""".strip()

logger = logging.getLogger(__name__)


def create_planner_agent(config: Config, tools: Tools) -> Agent:
    researcher_agent = create_researcher_agent(config, tools)
    expert_agent = create_expert_agent(config, tools)

    handoffs = [
        handoff(researcher_agent, input_filter=handoff_filters.remove_all_tools),
        handoff(expert_agent, input_filter=handoff_filters.remove_all_tools),
    ]

    return Agent(
        name="planner",
        instructions=PLANNER_INSTRUCTIONS,
        handoffs=handoffs,
    )
