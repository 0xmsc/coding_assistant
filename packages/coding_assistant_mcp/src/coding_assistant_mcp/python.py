from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.proc import start_process
from coding_assistant_mcp.utils import truncate_output
from coding_assistant_mcp.bg_tasks import manager

python_server = FastMCP()


async def execute(
    code: Annotated[str, "The Python code to execute."],
    timeout: Annotated[int, "The timeout for execution in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
    background: Annotated[bool, "If True, run the code in the background and return a task ID."] = False,
) -> str:
    """
    Execute the given Python code using uv run - and return combined stdout/stderr.

    The execution supports PEP 723 inline script metadata, allowing you to specify dependencies
    directly in the code block.
    """

    code = code.strip()

    try:
        # We use 'uv run -' to execute the code from stdin.
        handle = await start_process(
            args=["uv", "run", "-q", "-"],
            stdin_input=code,
        )

        if background:
            task_id = manager.register_task("python script", handle)
            return f"Task started in background with ID: {task_id}"

        finished = await handle.wait(timeout=timeout)

        if not finished:
            # Auto-backgrounding on timeout
            task_id = manager.register_task("python script (auto)", handle)
            return (
                f"Python script is taking longer than {timeout}s. "
                f"It has been moved to a background task with ID: {task_id}. "
                "You can check its status later using `bg_get_output`."
            )

        if handle.returncode != 0:
            return f"Exception (exit code {handle.returncode}):\n\n{handle.stdout}"

        return truncate_output(handle.stdout, truncate_at)

    except Exception as e:
        return f"Error executing script: {str(e)}"


python_server.tool(execute)
