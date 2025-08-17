import asyncio
import json
import logging
import re
import textwrap
from typing import List, Optional

from pydantic import BaseModel, Field

from coding_assistant.agents.callbacks import AgentCallbacks, NullCallbacks
from coding_assistant.agents.execution import run_agent_loop
from coding_assistant.agents.parameters import fill_parameters, format_parameters
from coding_assistant.agents.types import (
    Agent,
    AgentOutput,
    FinishTaskResult,
    ShortenConversationResult,
    TextResult,
    Tool,
)
from coding_assistant.config import Config
from coding_assistant.llm.model import complete
from coding_assistant.ui import UI, NullUI

logger = logging.getLogger(__name__)


async def _get_feedback(
    agent: Agent,
    config: Config,
    mcp_servers: list,
    enable_feedback_agent: bool,
    enable_user_feedback: bool,
    ui: UI,
    agent_callbacks: AgentCallbacks,
) -> str | None:
    if not agent.output:
        raise ValueError("Agent has no result to provide feedback on.")

    feedback = "Ok"

    if enable_feedback_agent:
        feedback_tool = FeedbackTool(config=config, mcp_servers=mcp_servers, agent_callbacks=agent_callbacks, ui=ui)
        formatted_parameters = textwrap.indent(format_parameters(agent.parameters), "  ")
        agent_feedback_result = await feedback_tool.execute(
            parameters={
                "description": agent.description,
                "parameters": "\n" + formatted_parameters,
                "result": agent.output.result,
                "summary": agent.output.summary,
            }
        )
        feedback = agent_feedback_result.content

    if enable_user_feedback:
        feedback = await ui.ask(f"Feedback for {agent.name}", default=feedback)

    return feedback if feedback != "Ok" else None


class LaunchOrchestratorAgentSchema(BaseModel):
    task: str = Field(description="The task to assign to the orchestrator agent.")
    summaries: List[str] = Field(
        default_factory=list,
        description="The past conversation summaries of the client and the agent.",
    )
    instructions: str | None = Field(
        default=None,
        description="Special instructions for the agent. The agent will do everything it can to follow these instructions. The orchestrator will forward relevant instructions to the other agents it launches.",
    )


class OrchestratorTool(Tool):
    def __init__(
        self, config: Config, mcp_servers: list, history: list | None, agent_callbacks: AgentCallbacks, ui: UI
    ):
        super().__init__()
        self._config = config
        self._mcp_servers = mcp_servers
        self._history = history
        self._agent_callbacks = agent_callbacks
        self._ui = ui

    def name(self) -> str:
        return "launch_orchestrator_agent"

    def description(self) -> str:
        return "Launch an orchestrator agent to accomplish a given task. The agent can delegate tasks to other agents where it sees fit. For bigger tasks, the orchestrator agent will make a plan with multiple milestones to tackle the task and ask the user whether it is okay to proceed with the plan."

    def parameters(self) -> dict:
        return LaunchOrchestratorAgentSchema.model_json_schema()

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
                AgentTool(self._config, self._mcp_servers, self._agent_callbacks, self._ui),
                AskClientTool(self._config.enable_ask_user, ui=self._ui),
                ExecuteShellCommandTool(self._config.shell_confirmation_patterns, ui=self._ui),
                FinishTaskTool(),
                ShortenConversation(),
            ],
            model=self._config.expert_model,
            tool_confirmation_patterns=self._config.tool_confirmation_patterns,
            feedback_function=lambda agent: _get_feedback(
                agent=agent,
                config=self._config,
                mcp_servers=self._mcp_servers,
                enable_feedback_agent=self._config.enable_feedback_agent,
                enable_user_feedback=self._config.enable_user_feedback,
                agent_callbacks=self._agent_callbacks,
                ui=self._ui,
            ),
        )

        try:
            output = await run_agent_loop(
                orchestrator_agent,
                self._agent_callbacks,
                self._config.shorten_conversation_at_tokens,
                self._config.no_truncate_tools,
                completer=complete,
                ui=self._ui,
            )
            self.summary = output.summary
            return TextResult(content=output.result)
        finally:
            self.history = orchestrator_agent.history


class LaunchAgentSchema(BaseModel):
    task: str = Field(description="The task to assign to the sub-agent.")
    expected_output: str = Field(
        description="The expected output to return to the client. This includes the content but also the format of the output (e.g. markdown).",
    )
    instructions: str | None = Field(
        default=None,
        description="Special instructions for the agent. The agent will do everything it can to follow these instructions.",
    )
    expert_knowledge: bool = Field(
        False,
        description="Should only be set to true when the task is difficult. When this is set to true, an expert-level agent will be used to work on the task.",
    )


class AgentTool(Tool):
    def __init__(self, config: Config, mcp_servers: list, agent_callbacks: AgentCallbacks, ui: UI):
        super().__init__()
        self._config = config
        self._mcp_servers = mcp_servers
        self._agent_callbacks = agent_callbacks
        self._ui = ui

    def name(self) -> str:
        return "launch_agent"

    def description(self) -> str:
        return "Launch a sub-agent to work on a given task. Examples for tasks are researching a topic or question, or developing a feature according to an implementation plan. The agent will refuse to accept any tasks that are not clearly defined and miss context. It needs to be clear what to do and how to do it using **only** the information given in the task description."

    def parameters(self) -> dict:
        return LaunchAgentSchema.model_json_schema()

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
                ExecuteShellCommandTool(self._config.shell_confirmation_patterns, ui=self._ui),
                AskClientTool(self._config.enable_ask_user, ui=self._ui),
                FinishTaskTool(),
                ShortenConversation(),
            ],
            model=self.get_model(parameters),
            tool_confirmation_patterns=self._config.tool_confirmation_patterns,
            feedback_function=lambda agent: _get_feedback(
                agent=agent,
                config=self._config,
                mcp_servers=self._mcp_servers,
                enable_feedback_agent=self._config.enable_feedback_agent,
                enable_user_feedback=self._config.enable_user_feedback,
                agent_callbacks=self._agent_callbacks,
                ui=self._ui,
            ),
        )

        output = await run_agent_loop(
            agent,
            self._agent_callbacks,
            self._config.shorten_conversation_at_tokens,
            self._config.no_truncate_tools,
            completer=complete,
            ui=self._ui,
        )
        return TextResult(content=output.result)


class AskClientSchema(BaseModel):
    question: str = Field(description="The question to ask the client.")
    default_answer: str | None = Field(default=None, description="A sensible default answer to the question.")


class AskClientTool(Tool):
    def __init__(self, enabled: bool, ui: UI):
        self.enabled = enabled
        self._ui = ui

    def name(self) -> str:
        return "ask_client"

    def description(self) -> str:
        return "Ask the client for input."

    def parameters(self) -> dict:
        return AskClientSchema.model_json_schema()

    async def execute(self, parameters: dict) -> TextResult:
        assert "question" in parameters

        if not self.enabled:
            return TextResult(
                "Client input is disabled for this session. Please continue as if the client had given the most sensible answer to your question."
            )

        question = parameters["question"]
        default_answer = parameters.get("default_answer")

        answer = await self._ui.ask(question, default=default_answer or "")
        return TextResult(content=str(answer))


class ExecuteShellCommandSchema(BaseModel):
    command: str = Field(description="The shell command to execute.")
    timeout: int | None = Field(default=None, description="The timeout for the command in seconds.")


class ExecuteShellCommandTool(Tool):
    def __init__(
        self,
        shell_confirmation_patterns: Optional[List[str]] = None,
        ui: UI | None = None,
    ):
        self._shell_confirmation_patterns = shell_confirmation_patterns or []
        self._ui = ui or NullUI()

    def name(self) -> str:
        return "execute_shell_command"

    def description(self) -> str:
        return "Execute a shell command and return the output. The command will be executed in bash."

    def parameters(self) -> dict:
        return ExecuteShellCommandSchema.model_json_schema()

    async def execute(self, parameters: dict) -> TextResult:
        assert "command" in parameters

        command = parameters["command"].strip()
        timeout = parameters.get("timeout", 30)

        for pattern in self._shell_confirmation_patterns:
            if re.search(pattern, command):
                question = f"Execute `{command}`?"
                answer = await self._ui.confirm(question)
                if not answer:
                    return TextResult(content="Command execution denied.")
                break

        logger.info(f"Executing shell command: `{command}`")

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            return TextResult(content=f"Command timed out after {timeout} seconds.")

        return TextResult(
            content=json.dumps(
                {
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "returncode": process.returncode,
                },
                indent=2,
            )
        )


class LaunchFeedbackAgentSchema(BaseModel):
    description: str = Field(description="The description of the agent that was working on the task.")
    parameters: str = Field(description="The parameters the agent was given for the task.")
    result: str = Field(description="The result of the agent.")
    summary: str | None = Field(default=None, description="A summary of the conversation with the client.")


class FeedbackTool(Tool):
    def __init__(self, config: Config, mcp_servers: list, agent_callbacks: AgentCallbacks, ui: UI):
        super().__init__()
        self._config = config
        self._mcp_servers = mcp_servers
        self._agent_callbacks = agent_callbacks
        self._ui = ui

    def name(self) -> str:
        return "launch_feedback_agent"

    def description(self) -> str:
        return "Launch a feedback agent that provides feedback on the output of another agent. This agent evaluates whether the result is acceptable for a given description, parameters, summary and feedback. The agent will evaluate the result as if it were a paying client. The feedback agent will thoroughly review every change that is described and will look at file system, git history, etc. as it deems necessary. If the result is acceptable, the feedback agent will call `finish_task` with the result being 'Ok'. Otherwise, it will output what is wrong with the result and how it needs to be improved."

    def parameters(self) -> dict:
        return LaunchFeedbackAgentSchema.model_json_schema()

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
                ExecuteShellCommandTool(self._config.shell_confirmation_patterns, ui=self._ui),
                FinishTaskTool(),
                ShortenConversation(),
            ],
            model=self._config.model,
            tool_confirmation_patterns=self._config.tool_confirmation_patterns,
            feedback_function=lambda agent: _get_feedback(
                agent=agent,
                config=self._config,
                mcp_servers=self._mcp_servers,
                enable_feedback_agent=False,
                enable_user_feedback=False,
                agent_callbacks=self._agent_callbacks,
                ui=self._ui,
            ),
        )

        output = await run_agent_loop(
            feedback_agent,
            self._agent_callbacks,
            self._config.shorten_conversation_at_tokens,
            self._config.no_truncate_tools,
            completer=complete,
            ui=self._ui,
        )
        return TextResult(content=output.result)


class FinishTaskSchema(BaseModel):
    result: str = Field(
        description="The result of the work on the task. The work of the agent is evaluated based on this result."
    )
    summary: str = Field(
        description="A concise summary of the conversation the agent and the client had. There should be enough context such that the work could be continued based on this summary. It should be able to evaluate your result using only your input parameters and this summary. That means that you need to include all of the user feedback you worked into your result.",
    )


class FinishTaskTool(Tool):
    def name(self) -> str:
        return "finish_task"

    def description(self) -> str:
        return "Signals that the assigned task is complete. This tool must be called eventually to terminate the agent's execution loop."

    def parameters(self) -> dict:
        return FinishTaskSchema.model_json_schema()

    async def execute(self, parameters) -> FinishTaskResult:
        return FinishTaskResult(
            result=parameters["result"],
            summary=parameters["summary"],
        )


class ShortenConversationSchema(BaseModel):
    summary: str = Field(description="A summary of the conversation so far.")


class ShortenConversation(Tool):
    def name(self) -> str:
        return "shorten_conversation"

    def description(self) -> str:
        return "Give the framework a summary of your conversation with the client so far. The work should be continuable based on this summary. This means that you need to include all the results you have already gathered so far. Additionally, you should include the next steps you had planned. This tool should only be called when the client tells you to call it."

    def parameters(self) -> dict:
        return ShortenConversationSchema.model_json_schema()

    async def execute(self, parameters) -> ShortenConversationResult:
        return ShortenConversationResult(summary=parameters["summary"])
