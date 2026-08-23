from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from coding_assistant.llm.types import TextToolResult, Tool
from coding_assistant.tools.process import execute_managed_process
from coding_assistant.tools.tasks import TaskManager


class ShellExecuteInput(BaseModel):
    command: str = Field(
        description=(
            "The shell command to execute. Do not include 'bash -c'. You can use shell features "
            "like pipes (|) and redirections (>, >>)."
        ),
    )
    timeout: int = Field(default=30, description="The timeout for the command in seconds.")
    truncate_at: int = Field(
        default=50_000,
        description="Maximum number of characters to return in stdout and stderr combined.",
    )
    background: bool = Field(
        default=False,
        description="If true, run the command in the background and return a task ID.",
    )


class ShellExecuteTool(Tool):
    """Execute shell commands through the local task manager."""

    def __init__(self, *, manager: TaskManager, process_env: dict[str, str] | None = None) -> None:
        self._manager = manager
        self._process_env = process_env

    def name(self) -> str:
        return "shell_execute"

    def description(self) -> str:
        return "Execute a shell command using bash and return combined stdout and stderr."

    def parameters(self) -> dict[str, Any]:
        return ShellExecuteInput.model_json_schema()

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        validated = ShellExecuteInput.model_validate(parameters)
        command = validated.command.strip()

        try:
            result = await execute_managed_process(
                args=["bash", "-c", command],
                task_name=f"shell: {command[:30]}...",
                register_task=self._manager.register_task,
                timeout=validated.timeout,
                truncate_at=validated.truncate_at,
                background=validated.background,
                env=self._process_env,
            )
            if result.status == "background":
                return TextToolResult(content=f"Task started in background with ID: {result.task_id}")
            if result.status == "timeout":
                return TextToolResult(
                    content=(
                        f"Command is taking longer than {validated.timeout}s. "
                        f"It continues in the background with Task ID: {result.task_id}. "
                        "You can check its status later using `tasks_get_output`."
                    ),
                )

            output = result.output
            if result.truncated:
                output += f"\n\n[Retained output available via `tasks_get_output(task_id={result.task_id})`]"

            if result.exit_code != 0:
                return TextToolResult(content=f"Exit code: {result.exit_code}.\n\n{output}")
            return TextToolResult(content=output)
        except Exception as exc:
            return TextToolResult(content=f"Error: {exc}")


def create_shell_tools(*, manager: TaskManager, process_env: dict[str, str] | None = None) -> list[Tool]:
    """Create the local shell execution tool."""
    return [ShellExecuteTool(manager=manager, process_env=process_env)]
