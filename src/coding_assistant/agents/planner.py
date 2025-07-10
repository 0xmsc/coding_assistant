import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List

from smolagents import CodeAgent, MultiStepAgent, Tool

from coding_assistant.config import Config

logger = logging.getLogger(__name__)

PLANNER_DESCRIPTION = """
Planner agent, which is responsible for planner an implementation task.
This agent is not a software architect.
Therefore, it should already be clear how to implement the task on a high level before handing a task to this agent.
Relevant paths, files, functions, etc. that are relevant to the task should be given to the planner agent.
It needs to know at which files it needs to look at, which functions are relevant, etc.
""".strip()


def create_planner_tools() -> List[Tool]:
    tools = []
    return tools


def create_planner_agent(config: Config) -> MultiStepAgent:
    return CodeAgent(
        model=config.model_factory(),
        tools=create_planner_tools(),
        name="Planner",
        description=PLANNER_DESCRIPTION,
    )
