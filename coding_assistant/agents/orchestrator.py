import logging
from dataclasses import dataclass
from pathlib import Path

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
from coding_assistant.logging import print_agent_progress
from coding_assistant.tools.research import research
from coding_assistant.tools.user import ask_user

ORCHESTRATOR_PROMPT = """
You are an orchestrator agent. Your responsibility is to orchestrate other agents to complete a task.
All other agents are available as tools.

Take the following steps to complete a task:

1. Use the research agent to gather information about the code base that are relevant to the task.
2. Split the task into multiple smaller tasks. Use the information gathered by the research agent to do this.
3. Use the planning agent to create an implementation plan for each of the smaller tasks.
4. Use the implementation agent to implement the plan for each of the smaller tasks.

Note that you don't have to follow these steps exactly. You can use the agents in any order you see fit.
You can also go back to any step at any time, if you need to.
This might be necessary if you encounter new information that changes your understanding of the task.
It might also be necessary if the output of one of the agents is not satisfactory.
It is your responsibility to make sure that the task is completed.

If something is unclear, you can ask the user for clarification.
This should be exceptional and not the norm.
Never stop working on the task to ask for clarification. Only use the ask_user tool when you're stuck.
You shall only stop when the task is done.

If you are missing an agent or a tool that would be needed to finish the task, output a description of what you would have done if the agent was available and what agent or tool you would have needed.
""".strip()

console = Console()
logger = logging.getLogger(__name__)


def create_orchestrator_tools():
    tools = []
    tools.append(ask_user)
    tools.append(research)
    return tools


def create_orchestrator_agent():
    memory = MemorySaver()
    model = ChatOpenAI(model_name="gpt-4o")
    tools = create_orchestrator_tools()
    return create_react_agent(model, tools, checkpointer=memory, prompt=ORCHESTRATOR_PROMPT)


def run_orchestrator_agent(task: str):
    agent = create_orchestrator_agent()
    return run_agent(agent, task, name="Orchestrator")
