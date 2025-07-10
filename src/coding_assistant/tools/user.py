from langchain_core.tools import tool
from typing import Annotated

from rich.prompt import Prompt
from coding_assistant.agents.researcher import run_researcher_agent


@tool
def ask_user(question: str) -> str:
    """
    Ask the user a question.
    """
    return Prompt.ask(question)
