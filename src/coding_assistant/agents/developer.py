import logging
from typing import Annotated

# Import necessary components from agents SDK
from agents import Agent, Handoff, Tool, handoff
from agents.extensions import handoff_filters

from coding_assistant.agents.researcher import create_researcher_tool
from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

DEVELOPER_INSTRUCTIONS = """
You are a Developer agent. Your responsibility is to execute implementation plans precisely.
Do not deviate from the provided plan or make architectural decisions.
You will receive detailed instructions for a task; execute them exactly as specified.
Use the file system tools provided to make the necessary code changes.
If you need clarification or information not present in the plan, use the researcher tool.
""".strip()


def create_developer_agent(config: Config, tools: Tools) -> Agent:
    return Agent(
        name="developer",
        instructions=DEVELOPER_INSTRUCTIONS,
        tools=[
            create_researcher_tool(config, tools),
        ],
        mcp_servers=tools.mcp_servers,
        model=config.model_factory(),
    )


def create_developer_tool(config: Config, tools: Tools) -> Tool:
    return create_developer_agent(config, tools).as_tool(
        "developer_tool",
        tool_description="Execute implementation plans precisely and make required code changes.",
    )
