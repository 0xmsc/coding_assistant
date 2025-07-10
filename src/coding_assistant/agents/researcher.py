import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from langchain_core.callbacks import BaseCallbackHandler

from coding_assistant.agents.agents import create_agent, create_context_prunning_prompt_function, run_agent
from coding_assistant.agents.expert import do_expert_analysis
from coding_assistant.agents.prompt import COMMON_AGENT_PROMPT
from coding_assistant.config import get_global_config
from coding_assistant.logging import print_agent_progress
from coding_assistant.tools.file import read_only_file_tools
from coding_assistant.tools.notebook import get_notebook_tools

RESEARCHER_PROMPT = f"""
You are a researcher agent. Your responsibility is to answer the question you're given.
Use the tools at your disposal to find the answer.

{COMMON_AGENT_PROMPT}

Note that you can't answer questions on how to implement something.
You can only give information about the current code base, not plan how to change it.
If someone asks you how to implement something, you should reject the task.

You always need to answer the researcher question tailored to the current code base.
It never makes sense to give a general answer.
Always reference files, snippets, functions, concepts, etc. in the code base.
When you show a code snippet, also give the file name where it is located.

Your output should contain all relevant information that you gathered during the researcher process.
It is not possible to ask follow-up questions regarding your task.
The output should be self-contained and not require follow-up questions.
Especially put interesting files and code snippets that are relevant to the question in the output.

Do not reject tasks easily. Always try to find a satisfactory answer.
Only reject tasks if you can't find any relevant information in the code base.
Before you reject a task, be really sure that the information is not there.

In some cases, you might be asked to verify an implementation according to an implementation plan.
Immediately reject the task, if you don't have the full implementation plan.

If you are missing a tool that would be helpful for your research, please let the user know.
""".strip()

console = Console()
logger = logging.getLogger(__name__)


def create_researcher_tools():
    tools = []
    tools.extend(read_only_file_tools())
    tools.append(do_expert_analysis)
    tools.extend(get_notebook_tools())
    return tools


def run_researcher_agent(question: str, notebook: dict, ask_user_for_feedback=False):
    agent = create_agent(
        prompt=create_context_prunning_prompt_function(RESEARCHER_PROMPT),
        tools=create_researcher_tools(),
        model=get_global_config().model_factory(),
    )
    return run_agent(
        agent,
        question,
        notebook=notebook,
        name="Researcher",
        ask_user_for_feedback=ask_user_for_feedback,
    )


@tool
def research(question: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Research a question about the current code base.
    The output will be a detailed answer in markdown format.
    """
    return run_researcher_agent(question, notebook=state["notebook"])
