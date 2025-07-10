from coding_assistant.agents.prompt import COMMON_AGENT_PROMPT
from coding_assistant.config import get_global_config
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from coding_assistant.agents.agents import run_agent
from coding_assistant.tools.file import read_only_file_tools

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
    return tools


def create_expert_agent():
    memory = MemorySaver()
    model = get_global_config().reasoning_model_factory()
    return create_react_agent(model, create_expert_tools(), checkpointer=memory, prompt=EXPERT_PROMPT)


def run_expert_agent(task: str, ask_user_for_feedback: bool = False):
    agent = create_expert_agent()
    return run_agent(agent, task, name="Expert", ask_user_for_feedback=ask_user_for_feedback)


@tool
def do_expert_analysis(question: str) -> str:
    """
    Let a software engineering expert anylze and answer a question.
    Note that this has to be an exceptionally difficult question that requires expert level knowledge.
    Additionally, all required context for answering the question has to be provided in the question.
    """
    if not get_global_config().reasoning_model_factory:
        return "Expert is not available..."

    return run_expert_agent(question)
