from dataclasses import dataclass
import logging
from typing import Annotated
from rich.prompt import Prompt

from coding_assistant.agents.logic import Agent, Parameter, run_agent_loop
from coding_assistant.config import Config
from coding_assistant.tools import Tool, Tools

logger = logging.getLogger(__name__)


def fill_parameters(
    parameter_description: dict,
    parameter_values: dict,
) -> list[Parameter]:
    parameters = []

    required = set(parameter_description.get("required", []))

    for name, parameter in parameter_description["properties"].items():
        value = parameter_values.get(name)

        if not value:
            if name in required:
                raise RuntimeError(f"Parameter {name} is required but not provided.")
            else:
                continue

        parameters.append(
            Parameter(
                name=name,
                description=parameter["description"],
                value=value,
            )
        )

    return parameters


class OrchestratorTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "launch_orchestrator_agent"

    def description(self) -> str:
        return "Launch an orchestrator agent to coordinate other agents. It will maximize the time it will offload to other agents, as its time valuable and expensive."

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
        orchestrator_agent = Agent(
            name="Orchestrator",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._tools.mcp_servers,
            tools=[
                ResearchTool(self._config, self._tools),
                DevelopTool(self._config, self._tools),
                AskUserTool(),
            ],
            model=self._config.model,
        )

        return await run_agent_loop(orchestrator_agent, ask_for_feedback=True)


class ResearchTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "launch_research_agent"

    def description(self) -> str:
        return "Launch a research agent to gather information. This agent can also access the codebase."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The research question to answer.",
                },
                "expected_output": {
                    "type": "string",
                    "description": "The expected output to return to the client. This includes the content but also the format of the output (e.g. markdown).",
                },
            },
            "required": ["question", "expected_output"],
        }

    async def execute(self, parameters: dict) -> str:
        research_agent = Agent(
            name="Researcher",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._tools.mcp_servers,
            tools=[],
            model=self._config.model,
        )

        return await run_agent_loop(research_agent)


class DevelopTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "launch_developer_agent"

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
            },
            "required": ["implementation_plan"],
        }

    async def execute(self, parameters: dict) -> str:
        developer_agent = Agent(
            name="Developer",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._tools.mcp_servers,
            tools=[],
            model=self._config.model,
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
                },
                "default_answer": {
                    "type": "string",
                    "description": "A sensible default answer to the question.",
                },
            },
            "required": ["question"],
        }

    async def execute(self, parameters: dict) -> str:
        assert "question" in parameters
        question = parameters["question"]
        default_answer = parameters.get("default_answer")

        answer = Prompt.ask(question, default=default_answer)
        return answer
