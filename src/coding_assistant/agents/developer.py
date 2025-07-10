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
from coding_assistant.agents.prompt import COMMON_AGENT_PROMPT
from coding_assistant.agents.researcher import research
from coding_assistant.config import get_global_config
from coding_assistant.tools.file import all_file_tools, read_only_file_tools

DEVELOPER_PROMPT = f"""
You are an developer agent. Your responsibility is to carry out a given implementation plan.

{COMMON_AGENT_PROMPT}

Note that it is not your responsibility to plan the implementation.
It is also not your responsibility to make decisions about the software architecture.

You should receive very detailed instructions on how to implement the task.
If it is unclear on how exactly to implement the task, you should reject the task.

Note that it is your responsibility to implement the task as closely as possible to the given implementation plan.
Implementation of the task always means that you need to change files.
Again, you are responsible for changing the files on disk, and you need to use write_files.

Your output should be a detailed description of all the tasks that you have done.
Only explain the changes you have actually done to the code on the filesystem.
Use this step to check if you have adhered to the implementation plan.
If not, you should correct the implementation.
Note that you should only explain changes that you have actually written to disk.
""".strip()

console = Console()
logger = logging.getLogger(__name__)


def create_developer_tools():
    tools = []

    tools.extend(all_file_tools())
    tools.append(research)

    return tools


def create_developer_agent():
    memory = MemorySaver()
    model = get_global_config().model_factory()
    tools = create_developer_tools()
    return create_react_agent(model, tools, checkpointer=memory, prompt=DEVELOPER_PROMPT)


def run_developer_agent(plan: str):
    agent = create_developer_agent()
    return run_agent(agent, plan, name="Developer")


@tool
def develop(plan: str) -> str:
    """
    Implement an imeplementation plan.
    The plan needs to be a detailed plan in markdown format.
    The output will be a detailed description of what has been implemented in markdown format.
    """
    return run_developer_agent(plan)
