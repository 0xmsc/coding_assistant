from coding_assistant.config import get_global_config
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from coding_assistant.agents.agents import run_agent

EXPERT_PROMPT = """
You are an expert agent. Your responsibility is to deal with exceptional tasks or queries...
""".strip()


def create_expert_tools():
    # Based on hypothetical tools similar to other agents
    tools = []
    # tools.extend(possibly_some_tools())
    # tools.append(some_other_tool)
    return tools


def create_expert_agent():
    memory = MemorySaver()
    model = get_global_config().reasoning_model_factory()
    return create_react_agent(model, create_expert_tools(), checkpointer=memory, prompt=EXPERT_PROMPT)


def is_exceptional(task: str) -> bool:
    # Example logic for determining exceptional tasks
    return "urgent" in task or "complex" in task


def run_expert_agent(task: str):
    if is_exceptional(task):
        agent = create_expert_agent()
        return run_agent(agent, task, name="Expert")
    else:
        return "Task is not considered exceptional for expert intervention."
