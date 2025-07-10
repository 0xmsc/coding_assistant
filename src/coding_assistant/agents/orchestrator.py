import logging
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.agents.developer import create_developer_agent
from coding_assistant.agents.planner import create_planner_agent
from coding_assistant.agents.researcher import create_researcher_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

ORCHESTRATOR_INSTRUCTIONS = """
You are an Orchestrator agent. Your goal is to coordinate other specialized agents to efficiently complete complex tasks.
""".strip()


def create_orchestrator_agent(config: Config, tools: Tools) -> Agent:
    planner_agent = create_planner_agent(config, tools)
    researcher_agent = create_researcher_agent(config, tools)
    developer_agent = create_developer_agent(config, tools)

    handoffs = [
        handoff(planner_agent, input_filter=handoff_filters.remove_all_tools),
        handoff(researcher_agent, input_filter=handoff_filters.remove_all_tools),
        handoff(developer_agent, input_filter=handoff_filters.remove_all_tools),
    ]

    return Agent(
        name="orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        tools=tools.sequential_thinking_tools,
        handoffs=handoffs,
    )
