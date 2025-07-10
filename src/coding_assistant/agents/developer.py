import logging
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.agents.researcher import create_researcher_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

DEVELOPER_INSTRUCTIONS = """
You are a Developer agent. Your responsibility is to execute implementation plans precisely.
Do not deviate from the provided plan or make architectural decisions.
You will receive detailed instructions for a task; execute them exactly as specified.
Use the file system tools provided to make the necessary code changes.
If you need clarification or information not present in the plan, hand off to the Researcher agent.
""".strip()


def create_developer_agent(config: Config, tools: Tools) -> Agent:
    researcher_agent = create_researcher_agent(config, tools)

    handoffs = [
        handoff(researcher_agent, input_filter=handoff_filters.remove_all_tools),
    ]

    return Agent(
        name="developer",
        instructions=DEVELOPER_INSTRUCTIONS,
        handoffs=handoffs,
        model=config.model_factory(),
    )
