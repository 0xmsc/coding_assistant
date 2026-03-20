import logging
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from coding_assistant.llm.types import Tool
from coding_assistant.tool_results import TextResult

logger = logging.getLogger(__name__)


class LaunchAgentSchema(BaseModel):
    task: str = Field(description="The task to assign to the agent.")
    expected_output: str | None = Field(
        default=None,
        description=(
            "The expected output to return to the client. This includes the content but also "
            "the format of the output (e.g. markdown)."
        ),
    )
    instructions: str | None = Field(
        default=None,
        description=(
            "Special instructions for the agent. The agent will do everything it can to follow these "
            "instructions. If appropriate, it can forward relevant instructions to the agents it launches."
        ),
    )
    expert_knowledge: bool = Field(
        False,
        description="Use an expert-level model when the task is difficult.",
    )


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
        return (
            "Call another tool and redirect its output to a file. Use this when the output of a tool is too large "
            "to be handled in the conversation or when you need to pipeline the result into another tool "
            "(for example: search -> file -> python)."
        )

    def parameters(self) -> dict[str, Any]:
        return RedirectToolCallSchema.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextResult:
        validated = RedirectToolCallSchema.model_validate(parameters)
        tool_name = validated.tool_name
        tool_args = validated.tool_args
        output_file = validated.output_file

        if tool_name == self.name():
            return TextResult(content="Error: Cannot call redirect_tool_call recursively.")

        target_tool = next((tool for tool in self._tools if tool.name() == tool_name), None)
        if target_tool is None:
            return TextResult(content=f"Error: Tool '{tool_name}' not found or cannot be redirected.")

        try:
            result = await target_tool.execute(tool_args)
            if not isinstance(result, TextResult):
                return TextResult(content=f"Error: Tool '{tool_name}' did not return a TextResult.")

            if output_file:
                path = Path(output_file)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(result.content)
                return TextResult(content=f"Tool '{tool_name}' executed. Output redirected to {output_file}")

            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
                tmp.write(result.content)
                return TextResult(
                    content=f"Tool '{tool_name}' executed. Output redirected to temporary file: {tmp.name}"
                )
        except Exception as exc:
            logger.exception("Error executing tool '%s' via redirect_tool_call", tool_name)
            return TextResult(content=f"Error executing tool '{tool_name}': {exc}")
