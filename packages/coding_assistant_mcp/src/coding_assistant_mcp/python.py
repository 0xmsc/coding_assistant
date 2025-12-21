from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.proc import start_process
from coding_assistant_mcp.utils import truncate_output
from coding_assistant_mcp.tasks import manager

python_server = FastMCP()


async def execute(
    code: Annotated[str, "The Python code to execute."],
    timeout: Annotated[int, "The timeout for execution in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
    background: Annotated[bool, "If True, run the code in the background and return a task ID."] = False,
) -> str:
    """
    Execute the given Python code using uv run - and return combined stdout/stderr.
    """

    code = code.strip()

    try:
        handle = await start_process(
            args=["uv", "run", "-q", "-"],
            stdin_input=code,
        )

        task_name = "python script"
        task_id = manager.register_task(task_name, handle)

        if background:
            return f"Task started in background with ID: {task_id}"

        finished = await handle.wait(timeout=timeout)

        if not finished:
            return (
                f"Python script is taking longer than {timeout}s. "
                f"It continues in the background with Task ID: {task_id}. "
                "You can check its status later using `tasks_get_output`."
            )

        output = handle.stdout
        stdout_text = truncate_output(output, truncate_at)

        if len(output) > truncate_at:
            stdout_text += f"\n\nFull output available via `tasks_get_output(task_id={task_id})`"

        if handle.returncode != 0:
            return f"Exception (exit code {handle.returncode}):\n\n{handle.stdout}"

        return stdout_text

    except Exception as e:
        return f"Error executing script: {str(e)}"


python_server.tool(execute)
