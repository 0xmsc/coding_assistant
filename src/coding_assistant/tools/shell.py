from __future__ import annotations

from pydantic import BaseModel, Field

from coding_assistant.tools.base import StructuredTool
from coding_assistant.tools.process import start_process, truncate_output
from coding_assistant.tools.tasks import TaskManager
from coding_assistant.llm.types import Tool


class ShellExecuteInput(BaseModel):
    command: str = Field(
        description=(
            "The shell command to execute. Do not include 'bash -c'. You can use shell features "
            "like pipes (|) and redirections (>, >>)."
        )
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


def create_shell_tools(*, manager: TaskManager) -> list[Tool]:
    """Create the local shell execution tool."""

    async def execute(validated: ShellExecuteInput) -> str:
        command = validated.command.strip()

        try:
            handle = await start_process(args=["bash", "-c", command])
            task_name = f"shell: {command[:30]}..."
            task_id = manager.register_task(task_name, handle)

            if validated.background:
                return f"Task started in background with ID: {task_id}"

            finished = await handle.wait(timeout=validated.timeout)
            if not finished:
                return (
                    f"Command is taking longer than {validated.timeout}s. "
                    f"It continues in the background with Task ID: {task_id}. "
                    "You can check its status later using `tasks_get_output`."
                )

            output = handle.stdout
            truncated = len(output) > validated.truncate_at
            output = truncate_output(output, validated.truncate_at)
            if truncated:
                output += f"\n\n[Full output available via `tasks_get_output(task_id={task_id})`]"

            if handle.exit_code != 0:
                return f"Exit code: {handle.exit_code}.\n\n{output}"
            return output
        except Exception as exc:
            return f"Error: {exc}"

    return [
        StructuredTool(
            name="shell_execute",
            description="Execute a shell command using bash and return combined stdout and stderr.",
            schema_model=ShellExecuteInput,
            handler=execute,
        )
    ]
