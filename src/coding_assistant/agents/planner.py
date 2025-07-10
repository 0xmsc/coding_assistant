import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from langchain_core.callbacks import BaseCallbackHandler

from coding_assistant.agents.agents import run_agent
from coding_assistant.agents.expert import do_expert_analysis
from coding_assistant.agents.prompt import COMMON_AGENT_PROMPT
from coding_assistant.agents.researcher import research
from coding_assistant.config import get_global_config
from coding_assistant.tools.file import read_only_file_tools

PLANNER_PROMPT = f"""
You are an planner agent. Your responsibility is to plan an implementation task.

{COMMON_AGENT_PROMPT}

The task should be sufficiently small for you to be able to plan it.
If you think the task is too big, reject the task and give a reason why.

Your responsibility is only to plan the implementation of a task. 
You are not responsible for the implementation itself.
Use the tools at your disposal.

Note that you should receive a high level description of the task.
It should include all the necessary context, like which files, functions, etc. to look at.
If you are missing any high-level information, you should reject the task, and give a reason why.

If you need more information, use the research tool.
Your output should be very detailed description in markdown on how to implement the given task.
It should be suitable for a junior engineer to implement the task.
That means that you should explain every step that is needed, but not necessarily every line that needs to be changed.
If you give an example code snippet, it should be clear in which file it should be placed.
It should also be very clear if a new file should be created, or code is to be added into an existing file.
""".strip()

console = Console()
logger = logging.getLogger(__name__)


def create_planner_tools():
    tools = []
    tools.extend(read_only_file_tools())
    tools.append(research)
    tools.append(do_expert_analysis)
    return tools


def create_planner_agent():
    memory = MemorySaver()
    model = get_global_config().model_factory()
    tools = create_planner_tools()
    return create_react_agent(model, tools, checkpointer=memory, prompt=PLANNER_PROMPT)


def run_planner_agent(task: str):
    agent = create_planner_agent()
    return run_agent(agent, task, name="planner")


@tool
def plan(task: str) -> str:
    """
    Plan an implementation.
    The task should contain all the necessary context, including which files, functions, etc. to look at.
    The output will be a detailed implementation plan in markdown format.
    """
    return run_planner_agent(task)
