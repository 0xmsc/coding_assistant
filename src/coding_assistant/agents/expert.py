import logging
from typing import Annotated, List

from smolagents import CodeAgent, MultiStepAgent, Tool, tool, InjectedState

from coding_assistant.config import Config, get_global_config
from coding_assistant.tools.file import read_only_file_tools
from coding_assistant.tools.notebook import get_notebook_tools

logger = logging.getLogger(__name__)

EXPERT_DESCRIPTION = """
Expert agent, which is responsible for dealing with exceptional tasks or queries.
This agent is expected to have expert level knowledge in software engineering and related fields.
If the question does not require expert level knowledge, the agent should reject the task immediately.
Additionally, the agent should reject the question if not all necessary context is provided.
""".strip()


def create_expert_tools() -> List[Tool]:
    tools = []
    tools.extend(read_only_file_tools())
    tools.extend(get_notebook_tools())
    return tools


def create_expert_agent(config: Config) -> MultiStepAgent:
    return CodeAgent(
        model=config.expert_model_factory(),
        tools=create_expert_tools(),
        name="Expert",
        description=EXPERT_DESCRIPTION,
    )


@tool
def do_expert_analysis(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Let a software engineering expert analyze and answer a question.
    Note that this has to be an exceptionally difficult question that requires expert level knowledge.
    Additionally, all required context for answering the question has to be provided in the question.
    """
    if not get_global_config().expert_model_factory:
        return "Expert is not available..."

    notebook = state["notebook"]
    expert_agent = create_expert_agent(get_global_config())
    return expert_agent.run(question, notebook=notebook)
