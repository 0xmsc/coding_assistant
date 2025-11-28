from __future__ import annotations

import asyncio
import io
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.utils import truncate_output

python_server = FastMCP()


def _execute_code(code: str) -> str:
    out_buf = io.StringIO()
    ns: dict[str, object] = {}
    with redirect_stdout(out_buf):
        with redirect_stderr(sys.stdout):
            try:
                exec(code, ns, ns)
                return out_buf.getvalue()
            except BaseException:
                tb = traceback.format_exc()
                return f"Exception:\n\n{out_buf.getvalue()}{tb}"


async def execute(
    code: Annotated[str, "The Python code to execute."],
    timeout: Annotated[int, "The timeout for execution in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 5_000,
) -> str:
    """Execute the given Python code using exec and return combined stdout/stderr."""

    code = code.strip()
    loop = asyncio.get_running_loop()

    try:
        fut = loop.run_in_executor(None, _execute_code, code)
        result = await asyncio.wait_for(fut, timeout=timeout)
    except asyncio.TimeoutError:
        return f"Command timed out after {timeout} seconds."

    result = truncate_output(result, truncate_at)

    return result


python_server.tool(execute)
