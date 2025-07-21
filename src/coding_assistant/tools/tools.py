import asyncio
import json
import logging
import subprocess
import textwrap
from dataclasses import dataclass, field
from typing import Annotated, Optional

from rich.prompt import Prompt

from coding_assistant.agents.execution import run_agent_loop
from coding_assistant.agents.types import (
    Agent,
    AgentOutput,
    Tool,
    TextResult,
    FinishTaskResult,
    ShortenConversationResult,
    ToolResult,
)
from coding_assistant.agents.parameters import fill_parameters, format_parameters, Parameter
from coding_assistant.agents.callbacks import AgentCallbacks, NullCallbacks
from coding_assistant.config import Config

logger = logging.getLogger(__name__)


async def _get_feedback(
    agent: Agent,
    config: Config,
    mcp_servers: list,
    ask_user_for_feedback: bool,
    ask_agent_for_feedback: bool,
    agent_callbacks: Optional[AgentCallbacks] = None,
) -> str | None:
    if not agent.output:
        raise ValueError("Agent has no result to provide feedback on.")

    if ask_agent_for_feedback:
        feedback_tool = FeedbackTool(config, mcp_servers, agent_callbacks)
        formatted_parameters = textwrap.indent(format_parameters(agent.parameters), "  ")
        agent_feedback_result = await feedback_tool.execute(
            parameters={
                "description": agent.description,
                "parameters": "\n" + formatted_parameters,
                "result": agent.output.result,
                "summary": agent.output.summary,
                "feedback": agent.output.feedback,
            }
        )
        agent_feedback = agent_feedback_result.content
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
        mcp_servers: list | None = None,
        history: list | None = None,
        agent_callbacks: Optional[AgentCallbacks] = None,
    ):
        self._config = config
        self._mcp_servers = mcp_servers or []
        self._history = history
        self._agent_callbacks = agent_callbacks or NullCallbacks()

    def name(self) -> str:
        return "launch_orchestrator_agent"

    def description(self) -> str:
        return "Launch an orchestrator agent to accomplish a given task. The agent can delegate tasks to other agents where it sees fit. For bigger tasks, the orchestrator agent will make a plan with multiple milestones to tackle the task and ask the user whether it is okay to proceed with the plan. Additionally, the orchestrator will ask the user whether it should continue on the current path or not after completion of each milestone."

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
                    "description": "Special instructions for the agent. The agent will do everything it can to follow these instructions. The orchestrator will forward relevant instructions to the other agents it launches.",
                },
            },
            "required": ["task"],
        }

    async def execute(self, parameters: dict) -> TextResult:
        orchestrator_agent = Agent(
            name="Orchestrator",
            history=self._history or [],
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._mcp_servers,
            tools=[
                AgentTool(self._config, self._mcp_servers, self._agent_callbacks),
                AskClientTool(),
                ExecuteShellCommandTool(),
                FinishTaskTool(),
                ShortenConversation(),
            ],
            model=self._config.expert_model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._mcp_servers,
                ask_user_for_feedback=self._config.enable_user_feedback,
                ask_agent_for_feedback=self._config.enable_feedback_agent,
                agent_callbacks=self._agent_callbacks,
            ),
        )

        try:
            output = await run_agent_loop(
                orchestrator_agent, self._agent_callbacks, self._config.shorten_conversation_at_tokens
            )
            self.summary = output.summary
        finally:
            self.history = orchestrator_agent.history

        return TextResult(content=output.result)


class AgentTool(Tool):
    def __init__(
        self, config: Config, mcp_servers: list | None = None, agent_callbacks: Optional[AgentCallbacks] = None
    ):
        self._config = config
        self._mcp_servers = mcp_servers or []
        self._agent_callbacks = agent_callbacks or NullCallbacks()

    def name(self) -> str:
        return "launch_agent"

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
                    "description": "Should only be set to true when the task is difficult. When this is set to true, an expert-level agent will be used to work on the task.",
                },
            },
            "required": ["task", "expected_output"],
        }

    def get_model(self, parameters: dict) -> str:
        if parameters.get("expert_knowledge"):
            return self._config.expert_model
        return self._config.model

    async def execute(self, parameters: dict) -> TextResult:
        agent = Agent(
            name="Agent",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._mcp_servers,
            tools=[
                ExecuteShellCommandTool(),
                AskClientTool(),
                FinishTaskTool(),
                ShortenConversation(),
            ],
            model=self.get_model(parameters),
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._mcp_servers,
                ask_user_for_feedback=self._config.enable_user_feedback,
                ask_agent_for_feedback=self._config.enable_feedback_agent,
                agent_callbacks=self._agent_callbacks,
            ),
        )

        output = await run_agent_loop(agent, self._agent_callbacks, self._config.shorten_conversation_at_tokens)
        return TextResult(content=output.result)


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

    async def execute(self, parameters: dict) -> TextResult:
        assert "question" in parameters
        question = parameters["question"]
        default_answer = parameters.get("default_answer")
        answer = await asyncio.to_thread(Prompt.ask, question, default=default_answer)
        return TextResult(content=str(answer))


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

    async def execute(self, parameters: dict) -> TextResult:
        assert "command" in parameters

        command = parameters["command"]
        args = ["bash", "-c", command]
        result = await asyncio.to_thread(subprocess.run, args, capture_output=True, text=True)

        if result.returncode != 0:
            return TextResult(
                content=(
                    f"Command failed with error code {result.returncode}\n"
                    f"stdout: {result.stdout}\n"
                    f"stderr: {result.stderr}"
                )
            )

        return TextResult(content=result.stdout)


class FeedbackTool(Tool):
    def __init__(
        self, config: Config, mcp_servers: list | None = None, agent_callbacks: Optional[AgentCallbacks] = None
    ):
        self._mcp_servers = mcp_servers or []
        self._config = config
        self._agent_callbacks = agent_callbacks or NullCallbacks()

    def name(self) -> str:
        return "launch_feedback_agent"

    def description(self) -> str:
        return "Launch a feedback agent that provides feedback on the output of another agent. This agent evaluates whether the result is acceptable for a given description, parameters, summary and feedback. The agent will evaluate the result as if it were a paying client. The feedback agent will thorougly review every change that is described and will look at file system, git history, etc. as it deems necessary. If the result is acceptable, the feedback agent will call `finish_task` with the result being 'Ok'. Otherwise, it will output what is wrong with the result and how it needs to be improved."

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

    async def execute(self, parameters: dict) -> TextResult:
        feedback_agent = Agent(
            name="Feedback",
            description=self.description(),
            parameters=fill_parameters(
                parameter_description=self.parameters(),
                parameter_values=parameters,
            ),
            mcp_servers=self._mcp_servers,
            tools=[
                ExecuteShellCommandTool(),
                FinishTaskTool(),
                ShortenConversation(),
            ],
            model=self._config.model,
            feedback_function=lambda agent: _get_feedback(
                agent,
                self._config,
                self._mcp_servers,
                ask_user_for_feedback=False,
                ask_agent_for_feedback=False,
                agent_callbacks=self._agent_callbacks,
            ),
        )

        output = await run_agent_loop(
            feedback_agent, self._agent_callbacks, self._config.shorten_conversation_at_tokens
        )
        return TextResult(content=output.result)


class FinishTaskTool(Tool):
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
                    "description": "A summary of the feedback given by the client to the agent during the task. This can both be questions that were answered by the client, or feedback. It needs to be clear from this parameter why the result might not fit to initial task description.",
                },
            },
            "required": ["result", "summary"],
        }

    async def execute(self, parameters) -> FinishTaskResult:
        return FinishTaskResult(
            result=parameters["result"],
            summary=parameters["summary"],
            feedback=parameters.get("feedback"),
        )


class ShortenConversation(Tool):
    def name(self) -> str:
        return "shorten_conversation"

    def description(self) -> str:
        return "Give the framework a short, concise summary of your conversation with the client so far. The work should be continuable based on this summary. This means that you need to include all the results you have already gathered so far. Additionally, you should include the next steps you had planned. This tool should only be called when the client tells you to call it."

    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A summary of the conversation so far.",
                },
            },
            "required": ["summary"],
        }

    async def execute(self, parameters) -> ShortenConversationResult:
        return ShortenConversationResult(summary=parameters["summary"])
