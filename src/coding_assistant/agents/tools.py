import asyncio
import json
import logging
import subprocess
import textwrap
from dataclasses import dataclass, field
from typing import Annotated

from rich.prompt import Prompt

from coding_assistant.agents.logic import (
    Agent,
    Parameter,
    fill_parameters,
    format_parameters,
    run_agent_loop,
    AgentOutput,
    Tool,
)
from coding_assistant.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Tools:
    mcp_servers: list = field(default_factory=list)
    tools: list = field(default_factory=list)


async def _get_feedback(
    agent: Agent,
    config: Config,
    tools: Tools,
    ask_user_for_feedback: bool,
    ask_agent_for_feedback: bool,
) -> str | None:
    if not agent.output:
        raise ValueError("Agent has no result to provide feedback on.")

    if ask_agent_for_feedback:
        feedback_tool = FeedbackTool(config, tools)
        formatted_parameters = textwrap.indent(format_parameters(agent.parameters), "  ")
        agent_feedback = await feedback_tool.execute(
            parameters={
                "description": agent.description,
                "parameters": "\n" + formatted_parameters,
                "result": agent.output.result,
                "summary": agent.output.summary,
                "feedback": agent.output.feedback,
            }
        )
    else:
        agent_feedback = "Ok"

    if ask_user_for_feedback:
        feedback = await asyncio.to_thread(Prompt.ask, f"Feedback for {agent.name}", default=agent_feedback)
    else:
        feedback = agent_feedback

    if feedback == "Ok":
        return None
    else:
        return feedback


class OrchestratorTool(Tool):
    def __init__(
        self,
        config: Config,
        tools: Tools,
        history: list | None = None,
    ):
        self._config = config
        self._tools = tools
        self._history = history

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
                },
                "summaries": {
                    "type": "array",
                    "description": "The past conversation summaries of the client and the agent.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Special instructions for the agent. The agent will do everything it can to follow these instructions. The orchestrator will forward these instructions to the other agents it launches.",
                },
            },
            "required": ["task"],
        }

    async def execute(self, parameters: dict) -> str:
        orchestrator_agent = Agent(
            name="Orchestrator",
            history=self._history or [],
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._tools.mcp_servers,
            tools=[
                AgentTool(self._config, self._tools),
                AskClientTool(),
                ExecuteShellCommandTool(),
            ],
            model=self._config.expert_model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._tools,
                ask_user_for_feedback=not self._config.disable_user_feedback,
                ask_agent_for_feedback=not self._config.disable_feedback_agent,
            ),
        )

        try:
            output = await run_agent_loop(orchestrator_agent)
            self.summary = output.summary
        finally:
            self.history = orchestrator_agent.history

        return output.result


class AgentTool(Tool):
    def __init__(self, config: Config, tools: Tools):
        self._config = config
        self._tools = tools

    def name(self) -> str:
        return "launch_research_agent"

    def description(self) -> str:
        return "Launch a sub-agent to work on a given task. Examples for tasks are researching a topic or question, or developing a feature according to an implementation plan. The agent will refuse to accept any tasks that are not clearly defined and miss context. It needs to be clear what to do and how to do it using **only** the information given in the task description."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to assign to the sub-agent.",
                },
                "expected_output": {
                    "type": "string",
                    "description": "The expected output to return to the client. This includes the content but also the format of the output (e.g. markdown).",
                },
                "instructions": {
                    "type": "string",
                    "description": "Special instructions for the agent. The agent will do everything it can to follow these instructions.",
                },
                "expert_knowledge": {
                    "type": "boolean",
                    "description": "Should only be set to true when the task is extraordinarily difficult. When this is set to true, an expert-level agent will be used to work on the task.",
                },
            },
            "required": ["task", "expected_output"],
        }

    def get_model(self, parameters: dict) -> str:
        if parameters.get("expert_knowledge"):
            return self._config.expert_model
        return self._config.model

    async def execute(self, parameters: dict) -> str:
        research_agent = Agent(
            name="Agent",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._tools.mcp_servers,
            tools=[
                ExecuteShellCommandTool(),
                AskClientTool(),
            ],
            model=self.get_model(parameters),
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._tools,
                ask_user_for_feedback=not self._config.disable_user_feedback,
                ask_agent_for_feedback=not self._config.disable_feedback_agent,
            ),
        )

        output = await run_agent_loop(research_agent)
        return output.result


class AskClientTool(Tool):
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
                    "description": "The question to ask the client.",
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
            "- `eza` or `ls` for listing files in a directory\n"
            "- `git` for running git commands\n"
            "- `fd` or `find` for searching files\n"
            "- `rg` or `grep` for searching text in files\n"
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
        return "Launch a feedback agent that provides feedback on the output of another agent. This agent evaluates whether the result is acceptable for a given description, parameters, summary and feedback. If it is, the feedback agent will call `finish_task` with the result being 'Ok' and nothing else. If it is not, the feedback agent will output as a result what is wrong with the result and how it needs to be improved. The agent will evaluate the result as if it were a paying client. Would an average client be satisfied with the output? The feedback agent will thorougly review every change that is described and will look at file system, git history, etc. as it deems necessary."

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
                "result": {
                    "type": "string",
                    "description": "The result of the agent.",
                },
                "summary": {
                    "type": "string",
                    "description": "A summary of the conversation with the client.",
                },
                "feedback": {
                    "type": "string",
                    "description": "The feedback provided to the agent during the work on the task.",
                },
            },
            "required": ["description", "parameters", "result"],
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
            tools=[
                ExecuteShellCommandTool(),
            ],
            model=self._config.model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._tools,
                ask_user_for_feedback=False,
                ask_agent_for_feedback=False,
            ),
        )

        output = await run_agent_loop(feedback_agent)
        return output.result


class FinishTaskTool(Tool):
    def __init__(self, agent: "Agent"):
        self._agent = agent

    def name(self) -> str:
        return "finish_task"

    def description(self) -> str:
        return "Signals that the assigned task is complete. This tool must be called eventually to terminate the agent's execution loop."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "result": {
                    "type": "string",
                    "description": "The result of the work on the task. The work of the agent is evaluated based on this result.",
                },
                "summary": {
                    "type": "string",
                    "description": "A concise summary of the conversation the agent and the client had. There should be enough context such that the work could be continued based on this summary.",
                },
                "feedback": {
                    "type": "string",
                    "description": "A summary of the feedback given by the client to the agent during the task. This can both be questions that were answered by the client, or feedback. It needs to be clear from this parameter why the result might might not fit to initial task description.",
                },
            },
            "required": ["result", "summary"],
        }

    async def execute(self, parameters) -> str:
        self._agent.output = AgentOutput(
            result=parameters["result"],
            summary=parameters["summary"],
            feedback=parameters.get("feedback"),
        )
        return "Agent output set."
