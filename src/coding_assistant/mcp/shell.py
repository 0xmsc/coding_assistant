from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

from coding_assistant.mcp.proc import start_process
from coding_assistant.mcp.utils import truncate_output
from coding_assistant.mcp.tasks import TaskManager


def create_shell_server(manager: TaskManager) -> FastMCP:
    shell_server = FastMCP("Shell")

    @shell_server.tool()
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

            task_name = f"shell: {command[:30]}..."
            task_id = manager.register_task(task_name, handle)

            if background:
                return f"Task started in background with ID: {task_id}"

            finished = await handle.wait(timeout=timeout)

            if not finished:
                return (
                    f"Command is taking longer than {timeout}s. "
                    f"It continues in the background with Task ID: {task_id}. "
                    "You can check its status later using `tasks_get_output`."
                )

            output = handle.stdout
            truncated = len(output) > truncate_at
            output = truncate_output(output, truncate_at)

            if truncated:
                output += f"\n\n[Full output available via `tasks_get_output(task_id={task_id})`]"

            if handle.exit_code != 0:
                return f"Exit code: {handle.exit_code}.\n\n{output}"
            else:
                return output

        except Exception as e:
            return f"Error: {str(e)}"

    return shell_server
