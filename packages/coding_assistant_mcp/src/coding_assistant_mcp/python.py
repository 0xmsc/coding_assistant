from __future__ import annotations

import asyncio
from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.utils import truncate_output

python_server = FastMCP()


async def execute(
    code: Annotated[str, "The Python code to execute."],
    timeout: Annotated[int, "The timeout for execution in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
) -> str:
    """Execute the given Python code using uv run - and return combined stdout/stderr."""

    code = code.strip()

    # We use 'uv run -' to execute the code from stdin.
    # -q (quiet) suppresses uv's own output (like resolving dependencies)
    # so we only get the script's output.
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        "-q",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    try:
        stdout, _ = await asyncio.wait_for(process.communicate(input=code.encode()), timeout=timeout)
        result = stdout.decode()
    except asyncio.TimeoutError:
        try:
            process.kill()
        except OSError:
            pass
        return f"Command timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing script: {str(e)}"

    if process.returncode != 0:
        return f"Exception (exit code {process.returncode}):\n\n{result}"

    result = truncate_output(result, truncate_at)

    return result


python_server.tool(execute)
