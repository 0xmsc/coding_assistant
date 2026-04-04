from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from pydantic import BaseModel, Field

from coding_assistant.llm.types import (
    CompactConversationResult,
    TextToolResult,
    Tool,
    ToolDefinition,
    ToolResult,
)


class CompactConversationSchema(BaseModel):
    summary: str = Field(description="A summary of the conversation so far.")


class CompactConversationTool(Tool):
    def name(self) -> str:
        """Return the built-in tool name for conversation compaction."""
        return "compact_conversation"

    def description(self) -> str:
        """Explain when the agent should summarize and reset prior context."""
        return "Give the runtime a summary of your conversation with the client so far. The work should be continuable based on this summary. This means that you need to include all the results you have already gathered so far. Additionally, you should include the next steps you had planned. When the user tells you to call this tool, you must do so immediately! This can also be called when you have reached a milestone and you think that the context you've gathered so far will not be relevant anymore going forward."

    def parameters(self) -> dict[str, Any]:
        """Return the schema for the compaction summary payload."""
        return CompactConversationSchema.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> CompactConversationResult:
        """Validate the request and return a compaction result."""
        validated = CompactConversationSchema.model_validate(parameters)
        return CompactConversationResult(summary=validated.summary)


class RedirectToolCallSchema(BaseModel):
    tool_name: str = Field(description="The name of the tool to call.")
    tool_args: dict[str, Any] = Field(description="The arguments to pass to the tool.")
    output_file: str | None = Field(
        default=None,
        description="The path where the output should be written. If omitted, a temporary file will be created.",
    )


class RedirectToolCallTool(Tool):
    """Run another tool and persist its output to a file."""

    def __init__(
        self,
        *,
        tools: Sequence[ToolDefinition],
        execute_tool: Callable[[str, dict[str, Any]], Awaitable[ToolResult]],
    ) -> None:
        self._tools = list(tools)
        self._execute_tool = execute_tool

    def name(self) -> str:
        """Return the built-in tool name for redirected tool output."""
        return "redirect_tool_call"

    def description(self) -> str:
        """Explain when large tool output should be redirected to disk."""
        return (
            "Call another tool and redirect its output to a file. Use this when the output of a tool is too large "
            "to be handled in the conversation or when you need to pipeline the result into another tool "
            "(for example: search -> file -> python)."
        )

    def parameters(self) -> dict[str, Any]:
        """Return the schema for redirected tool-call requests."""
        return RedirectToolCallSchema.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        """Execute the target tool and write its text output to a file."""
        validated = RedirectToolCallSchema.model_validate(parameters)
        tool_name = validated.tool_name
        tool_args = validated.tool_args
        output_file = validated.output_file

        if tool_name == self.name():
            return TextToolResult(content="Error: Cannot call redirect_tool_call recursively.")

        target_tool = next((tool for tool in self._tools if tool.name() == tool_name), None)
        if target_tool is None:
            return TextToolResult(content=f"Error: Tool '{tool_name}' not found or cannot be redirected.")

        result = await self._execute_tool(tool_name, tool_args)

        if not isinstance(result, TextToolResult):
            return TextToolResult(
                content=f"Error: Tool '{tool_name}' cannot be redirected because it does not produce text output.",
            )

        if output_file:
            path = Path(output_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(result.content)
            return TextToolResult(content=f"Tool '{tool_name}' executed. Output redirected to {output_file}")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp.write(result.content)
            return TextToolResult(
                content=f"Tool '{tool_name}' executed. Output redirected to temporary file: {tmp.name}",
            )
