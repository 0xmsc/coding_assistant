from __future__ import annotations

import asyncio
from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.utils import truncate_output

shell_server = FastMCP()


async def execute(
    command: Annotated[str, "The shell command to execute. Do not include 'bash -c'."],
    timeout: Annotated[int, "The timeout for the command in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 5_000,
) -> str:
    """Execute a shell command using bash and return combined stdout/stderr."""
    command = command.strip()

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        # Properly terminate the process on timeout
        proc.terminate()
        try:
            # Give the process a grace period to terminate
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            # Force kill if graceful termination didn't work
            proc.kill()
            await proc.wait()
        return f"Command timed out after {timeout} seconds."

    if proc.returncode != 0:
        result = f"Returncode: {proc.returncode}\n\n{stdout.decode()}"
    else:
        result = stdout.decode()

    result = truncate_output(result, truncate_at)

    return result


shell_server.tool(execute)
