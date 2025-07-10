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
from langchain_community.tools import ShellTool

from coding_assistant.agents.agents import create_agent, create_context_prunning_prompt_function, run_agent
from coding_assistant.agents.developer import develop
from coding_assistant.agents.expert import do_expert_analysis
from coding_assistant.agents.planner import plan
from coding_assistant.agents.prompt import COMMON_AGENT_PROMPT
from coding_assistant.agents.researcher import research
from coding_assistant.config import get_global_config
from coding_assistant.tools.file import read_only_file_tools
from coding_assistant.tools.git_tools import get_git_tools
from coding_assistant.tools.notebook import get_notebook_tools
from coding_assistant.tools.user import ask_user

ORCHESTRATOR_PROMPT = f"""
You are an orchestrator agent. Your responsibility is to orchestrate other agents to complete a task.
All other agents are available as tools.

{COMMON_AGENT_PROMPT}

Take the following steps to complete a task:

1. Use the research agent to gather information about the code base that are relevant to the task.
2. Split the task into multiple smaller implementation tasks.

For each of the smaller tasks, do the following:

1. Use the planning agent to create an implementation plan for each of the implementation tasks.
2. Use the developer agent to implement the plan for each of the smaller tasks.
3. Ask a research agent to verify that the changes are correct, according to the implementation plan and the output of the developer agent.

Note that you don't have to follow these steps exactly. You can use the agents in any order you see fit.
You can also go back to any step at any time, if you need to.
This might be necessary if you encounter new information that changes your understanding of the task.
It might also be necessary if the output of one of the agents is not satisfactory.
It is your responsibility to make sure that the task is completed.

Note that the planning agent is not a software architect.
Therefore, it should already be clear how to implement the task on a high level before handing it to the planning agent.
The planning agent can come up with a detailed plan on how to implement the task, like what functions to create, what classes to use, etc.
Give relevant paths, files, functions, etc. that are relevant to the task to the planning agent.
The planning agent is not supposed to start again from scratch. Give it all the necessary context from the research step.
It needs to know at which files it needs to look at, which functions are relevant, etc.

Note that the developer agent needs a very detailed plan to be able to implement the task.
Think of the agent as a junior engineer that needs to be told exactly what to do.

Note that the agents do not have access to each other's output.
You will have to provide ALL the necessary context to the agents via their task.
For example, you need to forward the full results of your research to the planning agents.
Another example would be that you need to fully forward the implementation plan from the planning agent to the developer agent.
Please also forward the full output of the developer agent to the research agent for verification.
In short it is of utmost importance that you provide all the necessary context to the agents.
It is better to provide too much context than too little.

If something is unclear, you can ask the user for clarification.
This should be exceptional and not the norm.
Never stop working on the task to ask for clarification. Only use the ask_user tool when you're stuck.
You shall only stop when the full task is done.

You should ask the user for clarifcation once you have an implementation plan.
Before you give it to the developer agent, you should ask the user if the plan is correct.
""".strip()

console = Console()
logger = logging.getLogger(__name__)


def create_orchestrator_tools():
    tools = []
    tools.extend(read_only_file_tools())
    tools.append(ShellTool(ask_human_input=True))
    tools.append(ask_user)
    tools.append(research)
    tools.append(plan)
    tools.append(develop)
    tools.append(do_expert_analysis)
    tools.extend(get_notebook_tools())
    tools.extend(get_git_tools())
    return tools


def run_orchestrator_agent(task: str, ask_user_for_feedback: bool):
    agent = create_agent(
        prompt=create_context_prunning_prompt_function(ORCHESTRATOR_PROMPT),
        tools=create_orchestrator_tools(),
        model=get_global_config().model_factory(),
    )
    return run_agent(agent, task, name="Orchestrator", ask_user_for_feedback=ask_user_for_feedback)
