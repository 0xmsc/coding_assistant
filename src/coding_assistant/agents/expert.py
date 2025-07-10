from typing import Annotated
from coding_assistant.agents.prompt import COMMON_AGENT_PROMPT
from coding_assistant.config import get_global_config
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from coding_assistant.agents.agents import create_agent, run_agent
from coding_assistant.tools.file import read_only_file_tools
from coding_assistant.agents.agents import create_context_prunning_prompt_function
from coding_assistant.tools.notebook import get_notebook_tools

EXPERT_PROMPT = f"""
You are an expert agent. Your responsibility is to deal with exceptional tasks or queries.

{COMMON_AGENT_PROMPT}

You are expected to have expert level knowledge in software engineering and related fields.
If you deem the question to not require expert level knowledge, you should reject the task immediately and give a reason why.
Additionally, reject the question if not all necessary context is provided.
""".strip()


def create_expert_tools():
    tools = []
    tools.extend(read_only_file_tools())
    tools.extend(get_notebook_tools())
    return tools


def run_expert_agent(task: str, notebook: dict, ask_user_for_feedback: bool):
    agent = create_agent(
        prompt=create_context_prunning_prompt_function(EXPERT_PROMPT, system_message_type="developer"),
        tools=create_expert_tools(),
        model=get_global_config().reasoning_model_factory(),
    )
    return run_agent(agent, task, notebook=notebook, name="Expert", ask_user_for_feedback=ask_user_for_feedback)


@tool
def do_expert_analysis(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Let a software engineering expert anylze and answer a question.
    Note that this has to be an exceptionally difficult question that requires expert level knowledge.
    Additionally, all required context for answering the question has to be provided in the question.
    """
    if not get_global_config().reasoning_model_factory:
        return "Expert is not available..."

    return run_expert_agent(question, notebook=state["notebook"], ask_user_for_feedback=True)
