from langchain_core.tools import tool
from typing import Annotated

from coding_assistant.agents.planner import run_planner_agent
from coding_assistant.agents.researcher import run_research_agent


@tool
def research(question: str) -> str:
    """
    Research a question about the current code base.
    The output will be a detailed answer in markdown format.
    """
    return run_research_agent(question)


@tool
def plan(task: str) -> str:
    """
    Plan an implementation.
    The output will be a detailed implementation plan in markdown format.
    """
    return run_planner_agent(task)
