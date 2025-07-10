import logging
from typing import List

from smolagents import CodeAgent, MultiStepAgent, Tool

from coding_assistant.agents.planner import create_planner_agent
from coding_assistant.config import Config

logger = logging.getLogger(__name__)

ORCHESTRATOR_DESCRIPTION = """
Orchestrator agent, which is responsible for orchestrating other agents to complete a task.
""".strip()


def create_orchestrator_tools() -> List[Tool]:
    tools = []
    return tools


def create_orchestrator_agent(config: Config) -> MultiStepAgent:
    return CodeAgent(
        model=config.model_factory(),
        tools=create_orchestrator_tools(),
        managed_agents=[create_planner_agent(config)],
        name="Orchestrator",
        description=ORCHESTRATOR_DESCRIPTION,
        planning_interval=3,
    )
