import logging
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from langchain_core.callbacks import BaseCallbackHandler

from coding_assistant.logging import print_agent_progress

RESEARCH_PROMPT = """
You are a research agent. Your responsibility is to answer the question you're given.
Use the tools at your disposal to find the answer.

Your output should be markdown formatted.
It should contain all relevant information that you gathered during the research process.
Note that you're output should be consumable by another agent.
Because of that, it is not possible to ask further questions.
The output should be self-contained and not require follow-up questions.
""".strip()

console = Console()
logger = logging.getLogger(__name__)


def create_research_tools(working_directory: Path):
    tools = []

    tools.extend(
        FileManagementToolkit(
            root_dir=str(working_directory),
            selected_tools=["read_file", "list_directory"],
        ).get_tools()
    )

    # TODO: Only when web search enabled
    tools.append(TavilySearchResults())

    return tools


def create_research_agent(working_directory: Path):
    memory = MemorySaver()
    model = ChatOpenAI(model_name="gpt-4o")
    tools = create_research_tools(working_directory=working_directory)
    return create_react_agent(model, tools, checkpointer=memory)


def run_research_agent(question: str, working_directory: Path):
    console.print(Panel(f"Research: {question}", title="Task", border_style="green"))
    config = {"configurable": {"thread_id": "thread"}}
    input = {"messages": HumanMessage(content=question)}
    agent = create_research_agent(working_directory=working_directory)
    for chunk in agent.stream(input=input, config=config):
        print_agent_progress(chunk)
