from dataclasses import dataclass
import logging
from typing import Annotated

from coding_assistant.agents.logic import Agent, run_agent_loop
from coding_assistant.config import Config
from coding_assistant.tools import Tool, Tools

logger = logging.getLogger(__name__)


class OrchestratorTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "orchestrate"

    def description(self) -> str:
        return "Launch an orchestrator agent to coordinate other agents."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to assign to the orchestrator agent.",
                }
            },
            "required": ["task"],
        }

    async def execute(self, parameters: dict) -> str:
        assert "task" in parameters
        task = parameters["task"]
        assert isinstance(task, str), "Task must be a string"

        # Inlined create_orchestrator_agent
        orchestrator_agent = Agent(
            name="orchestrator",
            instructions="You are an Orchestrator agent. Your goal is to coordinate other specialized agents to efficiently complete complex tasks. Note that your time is quite valuable and expensive, so you should maximize the time you offload to other agents.",
            mcp_servers=self._tools.mcp_servers,
            tools=[
                ResearchTool(self._config, self._tools),
                DevelopTool(self._config, self._tools),
                AskUserTool(),
            ],
            model=self._config.model,
            task=task,
            history=[],
        )

        return await run_agent_loop(orchestrator_agent)


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
        question = parameters["question"]

        # Inlined create_researcher_agent
        research_agent = Agent(
            name="researcher",
            instructions="You are a research agent. Your goal is to gather information and provide insights.",
            mcp_servers=self._tools.mcp_servers,
            tools=[],
            model=self._config.model,
            task=question,
            history=[],
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
                },
                "expected_output": {
                    "type": "string",
                    "description": "The expected output to return to the user. This includes content but also the format of the output (e.g. markdown).",
                },
            },
            "required": ["implementation_plan"],
        }

    async def execute(self, parameters: dict) -> str:
        assert "implementation_plan" in parameters
        implementation_plan = parameters["implementation_plan"]

        # Inlined create_developer_agent
        developer_agent = Agent(
            name="developer",
            instructions="You are a developer agent. Your goal is to write code according to an implementation plan given to you.",
            mcp_servers=self._tools.mcp_servers,
            tools=[],
            model=self._config.model,
            task=implementation_plan,
            history=[],
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
        question = parameters["question"]

        answer = input(f"{question}\nAnswer: ")
        return answer.strip()
