from __future__ import annotations

from pydantic import BaseModel, Field

from coding_assistant.tools.base import StructuredTool
from coding_assistant.tools.process import start_process, truncate_output
from coding_assistant.tools.tasks import TaskManager
from coding_assistant.llm.types import Tool


class PythonExecuteInput(BaseModel):
    code: str = Field(description="The Python code to execute.")
    timeout: int = Field(default=30, description="The timeout for execution in seconds.")
    truncate_at: int = Field(
        default=50_000,
        description="Maximum number of characters to return in stdout and stderr combined.",
    )
    background: bool = Field(
        default=False,
        description="If true, run the code in the background and return a task ID.",
    )


def create_python_tools(*, manager: TaskManager) -> list[Tool]:
    """Create the local Python execution tool."""

    async def execute(validated: PythonExecuteInput) -> str:
        code = validated.code.strip()

        try:
            handle = await start_process(
                args=["uv", "run", "-q", "-"],
                stdin_input=code,
                env={},
            )
            task_id = manager.register_task("python script", handle)

            if validated.background:
                return f"Task started in background with ID: {task_id}"

            finished = await handle.wait(timeout=validated.timeout)
            if not finished:
                return (
                    f"Python script is taking longer than {validated.timeout}s. "
                    f"It continues in the background with Task ID: {task_id}. "
                    "You can check its status later using `tasks_get_output`."
                )

            output = handle.stdout
            stdout_text = truncate_output(output, validated.truncate_at)
            if len(output) > validated.truncate_at:
                stdout_text += f"\n\nFull output available via `tasks_get_output(task_id={task_id})`"

            if handle.exit_code != 0:
                return f"Exception (exit code {handle.exit_code}):\n\n{handle.stdout}"
            return stdout_text
        except Exception as exc:
            return f"Error executing script: {exc}"

    return [
        StructuredTool(
            name="python_execute",
            description=(
                "Execute Python code using `uv run -q -`. Supports multi-line scripts and PEP 723 inline "
                "dependency metadata."
            ),
            schema_model=PythonExecuteInput,
            handler=execute,
        )
    ]
