import asyncio
from dataclasses import dataclass
import subprocess
import json
import logging
import textwrap
from typing import Annotated
from rich.prompt import Prompt

from coding_assistant.agents.logic import (
    Agent,
    Parameter,
    format_parameters,
    run_agent_loop,
    fill_parameters,
)
from coding_assistant.config import Config
from coding_assistant.tools import Tool, Tools

logger = logging.getLogger(__name__)


async def _get_feedback(
    agent: Agent,
    config: Config,
    tools: Tools,
    ask_user_for_feedback: bool,
    ask_agent_for_feedback: bool,
) -> str | None:
    if not agent.result:
        raise ValueError("Agent has no result to provide feedback on.")

    if ask_agent_for_feedback:
        feedback_tool = FeedbackTool(config, tools)
        formatted_parameters = textwrap.indent(format_parameters(agent.parameters), "  ")
        formatted_instructions = textwrap.indent(agent.instructions, "  ")
        agent_feedback = await feedback_tool.execute(
            parameters={
                # Give the system message as the task.
                "description": agent.description,
                "instructions": "\n" + formatted_instructions,
                "parameters": "\n" + formatted_parameters,
                "output": agent.result,
            }
        )
    else:
        agent_feedback = "Ok"

    if ask_user_for_feedback:
        feedback = Prompt.ask(f"Feedback for {agent.name}", default=agent_feedback)
    else:
        feedback = agent_feedback

    if feedback == "Ok":
        return None
    else:
        return feedback


class OrchestratorTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "launch_orchestrator_agent"

    def description(self) -> str:
        return "Launch an orchestrator agent to accomplish a given task. The agent can delegate tasks to other agents where it sees fit."

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
                ExecuteShellCommandTool(),
            ],
            model=self._config.model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._tools,
                ask_user_for_feedback=not self._config.disable_user_feedback,
                ask_agent_for_feedback=not self._config.disable_feedback_agent,
            ),
            instructions=self._config.instructions or "No additional instructions provided.",
        )

        return await run_agent_loop(orchestrator_agent)


class ResearchTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "launch_research_agent"

    def description(self) -> str:
        return "Launch a research agent to gather information."

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
            tools=[
                ExecuteShellCommandTool(),
            ],
            model=self._config.model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._tools,
                ask_user_for_feedback=not self._config.disable_user_feedback,
                ask_agent_for_feedback=not self._config.disable_feedback_agent,
            ),
            instructions=self._config.instructions or "No additional instructions provided.",
        )

        return await run_agent_loop(research_agent)


class DevelopTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "launch_developer_agent"

    def description(self) -> str:
        return "Launch a developer agent to write code according to an implementation plan. The developer agent will refuse to accept any tasks that are not clearly defined and miss context. It needs to be clear what to do and how to do it using **only** the information given in the implementation plan."

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
            tools=[
                ExecuteShellCommandTool(),
            ],
            model=self._config.model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._tools,
                ask_user_for_feedback=not self._config.disable_user_feedback,
                ask_agent_for_feedback=not self._config.disable_feedback_agent,
            ),
            instructions=self._config.instructions or "No additional instructions provided.",
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
        answer = await asyncio.to_thread(Prompt.ask, question, default=default_answer)
        return str(answer)


class ExecuteShellCommandTool(Tool):
    def __init__(self):
        pass

    def name(self) -> str:
        return "execute_shell_command"

    def description(self) -> str:
        return (
            "Execute a shell command and return the output. The command will be executed in bash. Examples for commands are:\n"
            "- `exa` for listing files in a directory\n"
            "- `git` for running git commands\n"
            "- `fd` for searching files\n"
            "- `rg` for searching text in files\n"
            "- `gh` for interfacing with GitHub\n"
        )

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
            },
            "required": ["command"],
        }

    async def execute(self, parameters: dict) -> str:
        assert "command" in parameters

        command = parameters["command"]
        args = ["bash", "-c", command]
        result = await asyncio.to_thread(subprocess.run, args, capture_output=True, text=True)

        if result.returncode != 0:
            return (
                f"Command failed with error code {result.returncode}\n"
                f"stdout: {result.stdout}\n"
                f"stderr: {result.stderr}"
            )

        return result.stdout


class FeedbackTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._tools = tools
        self._config = config

    def name(self) -> str:
        return "launch_feedback_agent"

    def description(self) -> str:
        return "Launch a feedback agent that provides feedback on the output of another agent. This agent evaluates whether the output is acceptable for a given task. If it is, the feedback agent will finish its task with only the output 'Ok' and nothing else. If it is not, the feedback agent will output what is wrong with the output and how it needs to be improved. Note that you shall evaluate the output as if you were a paying client. Would an average client be satisfied with the output? If the output describes filesystem changes, the feedback agent will use the filesystem tools to read the files and check if the changes are as described. Additionally, the agent can use git tools can be used to verify changes to the repository."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the agent that was working on the task.",
                },
                "parameters": {
                    "type": "string",
                    "description": "The parameters the agent was given for the task.",
                },
                "instructions": {
                    "type": "string",
                    "description": "The instructions the agent was given for the task.",
                },
                "output": {
                    "type": "string",
                    "description": "The output of the agent.",
                },
            },
            "required": ["description", "parameters", "output"],
        }

    async def execute(self, parameters: dict) -> str:
        feedback_agent = Agent(
            name="Feedback",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._tools.mcp_servers,
            tools=[],
            model=self._config.model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._tools,
                ask_user_for_feedback=False,
                ask_agent_for_feedback=False,
            ),
            instructions=self._config.instructions or "No additional instructions provided.",
        )

        return await run_agent_loop(feedback_agent)
