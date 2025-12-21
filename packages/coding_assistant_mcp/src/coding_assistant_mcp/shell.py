from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.proc import start_process
from coding_assistant_mcp.utils import truncate_output
from coding_assistant_mcp.bg_tasks import manager

shell_server = FastMCP()


async def execute(
    command: Annotated[str, "The shell command to execute. Do not include 'bash -c'."],
    timeout: Annotated[int, "The timeout for the command in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
    background: Annotated[bool, "If True, run the command in the background and return a task ID."] = False,
) -> str:
    """Execute a shell command using bash and return combined stdout/stderr."""
    command = command.strip()

    try:
        handle = await start_process(args=["bash", "-c", command])

        if background:
            task_id = manager.register_task(f"shell: {command[:20]}...", handle)
            return f"Task started in background with ID: {task_id}"

        finished = await handle.wait(timeout=timeout)

        if not finished:
            # Auto-backgrounding on timeout
            task_id = manager.register_task(f"shell (auto): {command[:20]}...", handle)
            return (
                f"Command is taking longer than {timeout}s. "
                f"It has been moved to a background task with ID: {task_id}. "
                "You can check its status later using `bg_get_output`."
            )

        stdout_text = truncate_output(handle.stdout, truncate_at)

        if handle.returncode != 0:
            return f"Returncode: {handle.returncode}.\n\n{stdout_text}"
        else:
            return stdout_text

    except Exception as e:
        return f"Error: {str(e)}"


shell_server.tool(execute)
