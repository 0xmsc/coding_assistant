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
from coding_assistant.agents.researcher import research
from coding_assistant.config import get_global_config

DEVELOPER_PROMPT = """
You are an developer agent. Your responsibility is to carry out a given implementation plan.

Note that it is not your responsibility to plan the implementation.
It is also not your responsibility to make decisions about the software architecture.

You should receive very detailed instructions on how to implement the task.
If it is unclear on how exactly to implement the task, you should reject the task.

If you are missing an agent or a tool that would be helpful for your task, please let the user know.
""".strip()

console = Console()
logger = logging.getLogger(__name__)


def create_developer_tools():
    tools = []

    working_directory = get_global_config().working_directory
    tools.extend(
        FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=["read_file", "list_directory", "write_file"],
        ).get_tools()
    )

    tools.append(research)

    return tools


def create_developer_agent():
    memory = MemorySaver()
    model = ChatOpenAI(model_name="gpt-4o")
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
