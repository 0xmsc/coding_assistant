from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated
import json


@tool
def record_fact(key: str, fact: str, notebook: Annotated[dict, InjectedState]):
    """
    Record a fact in a notebook.
    """
    notebook[key] = fact


@tool
def get_fact(key: str, notebook: Annotated[dict, InjectedState]) -> str:
    """
    Retrieve a fact from a notebook.
    """
    return notebook[key]


@tool
def delete_fact(key: str, notebook: Annotated[dict, InjectedState]):
    """
    Delete a fact from a notebook.
    """
    del notebook[key]


@tool
def get_facts(notebook: Annotated[dict, InjectedState]) -> str:
    """
    Retrieve all facts from a notebook.
    """
    return json.dumps(notebook)


def get_notebook_tools():
    return [record_fact, get_fact, delete_fact, get_facts]
