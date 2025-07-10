import logging
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()


def create_research_tools(working_directory: Path):
    tools = []

    tools.extend(
        FileManagementToolkit(
            root_dir=context.working_directory,
            selected_tools=["read_file", "list_directoy"],
        ).get_tools()
    )

    # TODO: Only when web search enabled
    tools.append(TavilySearchResults())

    return tools


def create_research_agent(working_directory: Path):
    memory = MemorySaver()
    # TODO: Make more generic
    model = ChatOpenAI(model_name="gpt-4o")
    tools = create_research_tools()
    logging.debug(f"Creating research agent with tools {tools}")
    return create_react_agent(model, tools, checkpointer=memory)


def run_research_agent(task: str, working_directory: Path):
    console.print(Panel(f"Research: {task}", style="blue"))
    agent = create_research_agent(context)
    for chunk in agent.stream(task):
        logging.debug(f"Agent: {chunk}")
