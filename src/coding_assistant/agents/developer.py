import logging
from typing import Annotated, List

from smolagents import CodeAgent, MultiStepAgent, Tool, tool, InjectedState
from rich.console import Console
from langchain_community.tools import ShellTool

from coding_assistant.agents.researcher import research
from coding_assistant.config import Config, get_global_config
from coding_assistant.tools.file import all_file_tools
from coding_assistant.tools.notebook import get_notebook_tools

console = Console()
logger = logging.getLogger(__name__)

DEVELOPER_DESCRIPTION = """
Developer agent, which is responsible for carrying out implementation plans.
This agent is not responsible for planning implementations or making decisions about software architecture.
The agent should receive detailed instructions on how to implement a task, and execute those instructions.
If the implementation plan is unclear, the agent should reject the task.
""".strip()


def create_developer_tools() -> List[Tool]:
    tools = []
    tools.extend(all_file_tools())
    tools.append(research)
    tools.extend(get_notebook_tools())
    tools.append(ShellTool(ask_human_input=True))
    return tools


def create_developer_agent(config: Config) -> MultiStepAgent:
    return CodeAgent(
        model=config.model_factory(),
        tools=create_developer_tools(),
        name="Developer",
        description=DEVELOPER_DESCRIPTION,
    )


@tool
def develop(plan: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Start a developer agent to implement a given plan.
    The plan needs to be detailed and in markdown format.
    The output will be a detailed description of what has been implemented in markdown format.
    """
    notebook = state["notebook"]
    developer_agent = create_developer_agent(get_global_config())
    return developer_agent.run(plan, notebook=notebook)
