import logging
from typing import List

from smolagents import CodeAgent, MultiStepAgent, Tool, ToolCallingAgent

from coding_assistant.agents.developer import create_developer_agent
from coding_assistant.agents.planner import create_planner_agent
from coding_assistant.agents.researcher import create_researcher_agent
from coding_assistant.config import Config
from coding_assistant.tools import Tools

logger = logging.getLogger(__name__)

ORCHESTRATOR_DESCRIPTION = "Coordinates other agents to complete tasks efficiently."


def create_orchestrator_agent(config: Config, tools: Tools) -> MultiStepAgent:
    return CodeAgent(
        model=config.expert_model_factory(),
        tools=[*tools.sequential_thinking_tools],
        managed_agents=[
            create_planner_agent(config, tools),
            create_researcher_agent(config, tools),
        ],
        name="orchestrator",
        description=ORCHESTRATOR_DESCRIPTION,
        planning_interval=3,
    )
