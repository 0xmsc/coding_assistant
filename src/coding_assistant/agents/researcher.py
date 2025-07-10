import logging
from typing import Annotated, List

from smolagents import CodeAgent, MultiStepAgent, Tool, tool, InjectedState
from rich.console import Console

from coding_assistant.agents.expert import do_expert_analysis
from coding_assistant.config import Config, get_global_config
from coding_assistant.tools.file import read_only_file_tools
from coding_assistant.tools.notebook import get_notebook_tools

console = Console()
logger = logging.getLogger(__name__)

RESEARCHER_DESCRIPTION = """
Researcher agent, which is responsible for answering questions about the code base.
This agent cannot implement changes to the code base, but can provide detailed information about it.
The agent should always reference files, snippets, functions, concepts, etc. in the code base.
When showing a code snippet, the agent should also give the file name where it is located.
The output should be self-contained and not require follow-up questions.
""".strip()


def create_researcher_tools() -> List[Tool]:
    tools = []
    tools.extend(read_only_file_tools())
    tools.append(do_expert_analysis)
    tools.extend(get_notebook_tools())
    return tools


def create_researcher_agent(config: Config) -> MultiStepAgent:
    return CodeAgent(
        model=config.model_factory(),
        tools=create_researcher_tools(),
        name="Researcher",
        description=RESEARCHER_DESCRIPTION,
    )


@tool
def research(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Research a question about the current code base.
    The output will be a detailed answer in markdown format.
    """
    notebook = state["notebook"]
    researcher_agent = create_researcher_agent(get_global_config())
    return researcher_agent.run(question, notebook=notebook)
