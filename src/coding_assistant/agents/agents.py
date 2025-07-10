from dataclasses import dataclass
import logging
from typing import Annotated

from coding_assistant.agents.logic import Agent, run_agent_loop
from coding_assistant.config import Config
from coding_assistant.tools import Tool, Tools

logger = logging.getLogger(__name__)

ORCHESTRATOR_INSTRUCTIONS = """
You are an Orchestrator agent. Your goal is to coordinate other specialized agents to efficiently complete complex tasks.
""".strip()


def create_orchestrator_agent(task: str, config: Config, tools: Tools) -> Agent:
    return Agent(
        name="orchestrator",
        instructions=ORCHESTRATOR_INSTRUCTIONS,
        mcp_servers=tools.mcp_servers,
        tools=[
            ResearchTool(config, tools),
            AskUserTool(),
        ],
        model=config.model,
        task=task,
        history=[],
    )


def create_researcher_agent(question: str, config: Config, tools: Tools) -> Agent:
    return Agent(
        name="researcher",
        instructions="You are a research agent. Your goal is to gather information and provide insights.",
        mcp_servers=tools.mcp_servers,
        tools=[],
        model=config.model,
        task=question,
        history=[],
    )


class ResearchTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "do_research"

    def description(self) -> str:
        return "Launch a research agent to gather information. This agent can also access the codebase."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The research question to answer.",
                }
            },
            "required": ["question"],
        }

    async def execute(self, parameters: dict) -> str:
        assert "question" in parameters
        question = parameters.get("question")

        # Create a new research agent
        research_agent = create_researcher_agent(
            question=question,
            config=self._config,
            tools=self._tools,
        )

        return await run_agent_loop(research_agent)


class AskUserTool(Tool):
    def __init__(self):
        pass

    def name(self) -> str:
        return "ask_user"

    def description(self) -> str:
        return "Ask the user for input."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user.",
                }
            },
            "required": ["question"],
        }

    async def execute(self, parameters: dict) -> str:
        assert "question" in parameters
        question = parameters.get("question")

        answer = input(question)
        return answer
