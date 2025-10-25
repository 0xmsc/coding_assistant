import logging

from pydantic import BaseModel, Field

from coding_assistant.agents.callbacks import (
    AgentProgressCallbacks,
    AgentToolCallbacks,
    NullProgressCallbacks,
)
from coding_assistant.agents.execution import run_agent_loop
from coding_assistant.agents.parameters import Parameter, parameters_from_model
from coding_assistant.agents.types import (
    AgentContext,
    AgentDescription,
    AgentState,
    FinishTaskResult,
    ShortenConversationResult,
    TextResult,
    Tool,
)
from coding_assistant.config import Config
from coding_assistant.llm.model import complete
from coding_assistant.ui import UI, DefaultAnswerUI

logger = logging.getLogger(__name__)


class LaunchOrchestratorAgentSchema(BaseModel):
    task: str = Field(description="The task to assign to the orchestrator agent.")
    summaries: list[str] = Field(
        default_factory=list,
        description="The past conversation summaries of the client and the agent.",
    )
    instructions: str | None = Field(
        default=None,
        description="Special instructions for the agent. The agent will do everything it can to follow these instructions. The orchestrator will forward relevant instructions to the other agents it launches.",
    )


class OrchestratorTool(Tool):
    def __init__(
        self,
        config: Config,
        tools: list[Tool],
        history: list | None,
        agent_callbacks: AgentProgressCallbacks,
        ui: UI,
        tool_callbacks: AgentToolCallbacks,
    ):
        super().__init__()
        self._config = config
        self._tools = tools
        self._history = history
        self._agent_callbacks = agent_callbacks
        self._ui = ui
        self._tool_callbacks = tool_callbacks

    def name(self) -> str:
        return "launch_orchestrator_agent"

    def description(self) -> str:
        return "Launch an orchestrator agent to accomplish a given task."

    def parameters(self) -> dict:
        return LaunchOrchestratorAgentSchema.model_json_schema()

    async def execute(self, parameters: dict) -> TextResult:
        # Compose parameters with the tool description as a dedicated entry
        validated = LaunchOrchestratorAgentSchema.model_validate(parameters)
        params = [
            Parameter(
                name="description",
                description="The description of the agent's work and capabilities.",
                value=self.description(),
            ),
            *parameters_from_model(validated),
        ]

        desc = AgentDescription(
            name="Orchestrator",
            model=self._config.expert_model,
            parameters=params,
            tools=[
                FinishTaskTool(),
                ShortenConversation(),
                AgentTool(
                    self._config,
                    self._tools,
                    DefaultAnswerUI(),
                    NullProgressCallbacks(),
                    self._tool_callbacks,
                ),
                AskClientTool(self._config.enable_ask_user, ui=self._ui),
                *self._tools,
            ],
        )
        state = AgentState(history=self._history or [])

        try:
            ctx = AgentContext(desc=desc, state=state)
            await run_agent_loop(
                ctx,
                agent_callbacks=self._agent_callbacks,
                tool_callbacks=self._tool_callbacks,
                shorten_conversation_at_tokens=self._config.shorten_conversation_at_tokens,
                enable_user_feedback=self._config.enable_user_feedback,
                completer=complete,
                ui=self._ui,
                is_interruptible=True,
            )
            assert state.output is not None, "Agent did not produce output"
            self.summary = state.output.summary
            return TextResult(content=state.output.result)
        finally:
            self.history = state.history


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
    def __init__(
        self,
        config: Config,
        tools: list[Tool],
        ui: UI,
        agent_callbacks: AgentProgressCallbacks,
        tool_callbacks: AgentToolCallbacks,
    ):
        super().__init__()
        self._config = config
        self._tools = tools
        self._ui = ui
        self._agent_callbacks = agent_callbacks
        self._tool_callbacks = tool_callbacks

    def name(self) -> str:
        return "launch_agent"

    def description(self) -> str:
        return "Launch a sub-agent to work on a given task. The agent will refuse to accept any task that is not clearly defined and misses context. It needs to be clear what to do using **only** the information given in the task description."

    def parameters(self) -> dict:
        return LaunchAgentSchema.model_json_schema()

    def get_model(self, parameters: dict) -> str:
        if parameters.get("expert_knowledge"):
            return self._config.expert_model
        return self._config.model

    async def execute(self, parameters: dict) -> TextResult:
        validated = LaunchAgentSchema.model_validate(parameters)
        params = [
            Parameter(
                name="description",
                description="The description of the agent's work and capabilities.",
                value=self.description(),
            ),
            *parameters_from_model(validated),
        ]

        desc = AgentDescription(
            name="Agent",
            model=self.get_model(parameters),
            parameters=params,
            tools=[
                FinishTaskTool(),
                ShortenConversation(),
                *self._tools,
            ],
        )
        state = AgentState()
        ctx = AgentContext(desc=desc, state=state)

        await run_agent_loop(
            ctx,
            agent_callbacks=self._agent_callbacks,
            tool_callbacks=self._tool_callbacks,
            shorten_conversation_at_tokens=self._config.shorten_conversation_at_tokens,
            enable_user_feedback=False,
            completer=complete,
            is_interruptible=False,
            ui=self._ui,
        )
        assert state.output is not None, "Agent did not produce output"
        return TextResult(content=state.output.result)


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


class FinishTaskSchema(BaseModel):
    result: str = Field(
        description="The result of the work on the task. The work of the agent is evaluated based on this result."
    )
    summary: str = Field(
        description="A concise summary of the conversation the agent and the client had. The summary shall be a single paragraph, without line breaks. There should be enough context such that the work could be continued based on this summary. It should be possible to evaluate your result using only your input parameters and this summary. That means that you need to include all of the user feedback you worked into your result.",
    )


class FinishTaskTool(Tool):
    def name(self) -> str:
        return "finish_task"

    def description(self) -> str:
        return "Signals that the assigned task is complete. This tool must be called eventually to terminate the agent's execution loop. This tool shall not be called when there are still open questions for the client."

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
