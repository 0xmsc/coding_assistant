import logging
import tempfile
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import (
    BaseMessage,
    NullProgressCallbacks,
    ProgressCallbacks,
    Tool,
)
from coding_assistant.framework.agent import run_agent_loop
from coding_assistant.framework.parameters import Parameter, parameters_from_model
from coding_assistant.framework.types import (
    AgentContext,
    AgentDescription,
    AgentState,
    Completer,
)
from coding_assistant.framework.results import TextResult
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.ui import DefaultAnswerUI, UI

logger = logging.getLogger(__name__)


class LaunchAgentSchema(BaseModel):
    task: str = Field(description="The task to assign to the agent.")
    expected_output: str | None = Field(
        default=None,
        description="The expected output to return to the client. This includes the content but also the format of the output (e.g. markdown).",
    )
    instructions: str | None = Field(
        default=None,
        description="Special instructions for the agent. The agent will do everything it can to follow these instructions. If appropriate, the agent will forward relevant instructions to the other agents it launches.",
    )
    expert_knowledge: bool = Field(
        False,
        description="Should only be set to true when the task is difficult. When this is set to true, an expert-level agent will be used to work on the task.",
    )


class AskClientSchema(BaseModel):
    question: str = Field(description="The question to ask the client.")
    default_answer: str | None = Field(default=None, description="A sensible default answer to the question.")


class AskClientTool(Tool):
    def __init__(self, *, ui: UI):
        super().__init__()
        self._ui = ui

    def name(self) -> str:
        return "ask_client"

    def description(self) -> str:
        return "Ask the client for input."

    def parameters(self) -> dict[str, Any]:
        return AskClientSchema.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        validated = AskClientSchema.model_validate(parameters)
        answer = await self._ui.ask(validated.question, default=validated.default_answer)
        return TextResult(content=str(answer))


class RedirectToolCallSchema(BaseModel):
    tool_name: str = Field(description="The name of the tool to call.")
    tool_args: dict[str, Any] = Field(description="The arguments to pass to the tool.")
    output_file: str | None = Field(
        default=None,
        description="The path where the output should be written. If omitted, a temporary file will be created.",
    )


class RedirectToolCallTool(Tool):
    def __init__(self, *, tools: list[Tool]):
        super().__init__()
        self._tools = tools

    def name(self) -> str:
        return "redirect_tool_call"

    def description(self) -> str:
        return "Call another tool and redirect its output to a file. Use this when the output of a tool is too large to be handled in the conversation or when you need to pipeline the result into another tool (e.g., search -> file -> python)."

    def parameters(self) -> dict[str, Any]:
        return RedirectToolCallSchema.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        validated = RedirectToolCallSchema.model_validate(parameters)
        tool_name = validated.tool_name
        tool_args = validated.tool_args
        output_file = validated.output_file

        if tool_name == self.name():
            return TextResult(content="Error: Cannot call redirect_tool_call recursively.")

        target_tool = next((t for t in self._tools if t.name() == tool_name), None)
        if not target_tool:
            return TextResult(content=f"Error: Tool '{tool_name}' not found or cannot be redirected.")

        try:
            result = await target_tool.execute(tool_args)

            if not isinstance(result, TextResult):
                return TextResult(content=f"Error: Tool '{tool_name}' did not return a TextResult.")

            content = result.content

            if output_file:
                path = Path(output_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                return TextResult(content=f"Tool '{tool_name}' executed. Output redirected to {output_file}")
            else:
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
                    tmp.write(content)
                    return TextResult(
                        content=f"Tool '{tool_name}' executed. Output redirected to temporary file: {tmp.name}"
                    )

        except Exception as e:
            logger.exception(f"Error executing tool '{tool_name}' via redirect_tool_call")
            return TextResult(content=f"Error executing tool '{tool_name}': {e}")


class AgentTool(Tool):
    def __init__(
        self,
        *,
        model: str,
        expert_model: str,
        compact_conversation_at_tokens: int,
        enable_ask_user: bool,
        tools: list[Tool],
        ui: UI,
        progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
        tool_callbacks: ToolCallbacks = NullToolCallbacks(),
        name: str = "launch_agent",
        history: Sequence[BaseMessage] | None = None,
        completer: Completer | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._expert_model = expert_model
        self._compact_conversation_at_tokens = compact_conversation_at_tokens
        self._enable_ask_user = enable_ask_user
        self._tools = tools
        self._ui = ui
        self._progress_callbacks = progress_callbacks
        self._tool_callbacks = tool_callbacks
        self._name = name
        self._history = history
        self._completer = completer or openai_complete
        self.history: list[BaseMessage] = []
        self.summary: str = ""

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "Launch an agent to work on a given task. The agent will refuse to accept any task that is not clearly defined and misses context. It needs to be clear what to do using **only** the information given in the task description."

    def parameters(self) -> dict[str, Any]:
        return LaunchAgentSchema.model_json_schema()

    def get_model(self, parameters: dict[str, Any]) -> str:
        if parameters.get("expert_knowledge"):
            return self._expert_model
        return self._model

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
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
                AgentTool(
                    model=self._model,
                    expert_model=self._expert_model,
                    compact_conversation_at_tokens=self._compact_conversation_at_tokens,
                    enable_ask_user=self._enable_ask_user,
                    tools=self._tools,
                    ui=DefaultAnswerUI() if not self._enable_ask_user else self._ui,
                    progress_callbacks=NullProgressCallbacks(),
                    tool_callbacks=self._tool_callbacks,
                    completer=self._completer,
                ),
                *self._tools,
            ],
        )
        state = AgentState(history=list(self._history) if self._history is not None else [])
        ctx = AgentContext(desc=desc, state=state)

        try:
            await run_agent_loop(
                ctx,
                progress_callbacks=self._progress_callbacks,
                tool_callbacks=self._tool_callbacks,
                compact_conversation_at_tokens=self._compact_conversation_at_tokens,
                completer=self._completer,
                ui=self._ui,
            )
            assert state.output is not None, "Agent did not produce output"
            self.summary = state.output.summary
            return TextResult(content=state.output.result)
        finally:
            self.history = state.history
