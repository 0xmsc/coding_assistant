from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.proc import execute_process
from coding_assistant_mcp.utils import truncate_output

shell_server = FastMCP()


async def execute(
    command: Annotated[str, "The shell command to execute. Do not include 'bash -c'."],
    timeout: Annotated[int, "The timeout for the command in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
) -> str:
    """Execute a shell command using bash and return combined stdout/stderr."""
    command = command.strip()

    try:
        result = await execute_process(
            args=["bash", "-c", command],
            timeout=timeout,
        )

        stdout_text = truncate_output(result.stdout, truncate_at)

        if result.timed_out:
            return f"Command timed out after {timeout} seconds.\n\n{stdout_text}"
        elif result.returncode != 0:
            return f"Returncode: {result.returncode}.\n\n{stdout_text}"
        else:
            return stdout_text

    except Exception as e:
        return f"Error: {str(e)}"


shell_server.tool(execute)
