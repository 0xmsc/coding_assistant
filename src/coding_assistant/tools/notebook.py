from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
import json


@tool
def record_fact(key: str, fact: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Record a fact in the notebook.
    """
    notebook = state["notebook"]
    notebook[key] = fact
    return "Success"


@tool
def get_fact(key: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Retrieve a fact from the notebook.
    """
    notebook = state["notebook"]
    return notebook[key]


@tool
def delete_fact(key: str, state: Annotated[dict, InjectedState]) -> str:
    """
    Delete a fact from the notebook.
    """
    notebook = state["notebook"]
    del notebook[key]
    return "Success"


@tool
def get_facts(state: Annotated[dict, InjectedState]) -> str:
    """
    Retrieve all facts from the notebook.
    """
    notebook = state["notebook"]
    return json.dumps(notebook)


def get_notebook_tools():
    return [record_fact, get_fact, delete_fact, get_facts]
