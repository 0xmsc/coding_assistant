from dataclasses import dataclass
import logging
from typing import Annotated

from coding_assistant.agents.logic import Agent, run_agent_loop
from coding_assistant.config import Config
from coding_assistant.tools import Tool, Tools

logger = logging.getLogger(__name__)


def create_orchestrator_agent(task: str, config: Config, tools: Tools) -> Agent:
    return Agent(
        name="orchestrator",
        instructions="You are an Orchestrator agent. Your goal is to coordinate other specialized agents to efficiently complete complex tasks. Note that your time is quite valuable and expensive, so you should maximize the time you offload to other agents.",
        mcp_servers=tools.mcp_servers,
        tools=[
            ResearchTool(config, tools),
            DevelopTool(config, tools),
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


def create_developer_agent(task: str, config: Config, tools: Tools) -> Agent:
    return Agent(
        name="developer",
        instructions="You are a developer agent. Your goal is to write code according to an implementation_plan given to you.",
        mcp_servers=tools.mcp_servers,
        tools=[],
        model=config.model,
        task=task,
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


class DevelopTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "develop"

    def description(self) -> str:
        return "Launch a developer agent to write code according to an implementation plan."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "implementation_plan": {
                    "type": "string",
                    "description": "The implementation plan to follow.",
                }
            },
            "required": ["implementation_plan"],
        }

    async def execute(self, parameters: dict) -> str:
        assert "implementation_plan" in parameters
        implementation_plan = parameters.get("implementation_plan")

        # Create a new developer agent
        developer_agent = create_developer_agent(
            task=implementation_plan,
            config=self._config,
            tools=self._tools,
        )

        return await run_agent_loop(developer_agent)


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

        answer = input(f"{question}\nAnswer: ")
        return answer.strip()
