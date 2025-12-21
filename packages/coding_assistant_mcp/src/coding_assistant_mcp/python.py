from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.proc import execute_process
from coding_assistant_mcp.utils import truncate_output

python_server = FastMCP()


async def execute(
    code: Annotated[str, "The Python code to execute."],
    timeout: Annotated[int, "The timeout for execution in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
) -> str:
    """
    Execute the given Python code using uv run - and return combined stdout/stderr.

    The execution supports PEP 723 inline script metadata, allowing you to specify dependencies
    directly in the code block.

    Example:
    ```python
    # /// script
    # dependencies = ["requests"]
    # ///
    import requests
    print(requests.get("https://github.com").status_code)
    ```
    """

    code = code.strip()

    try:
        # We use 'uv run -' to execute the code from stdin.
        # -q (quiet) suppresses uv's own output (like resolving dependencies)
        # so we only get the script's output.
        result = await execute_process(
            args=["uv", "run", "-q", "-"],
            stdin_input=code,
            timeout=timeout,
        )

        if result.timed_out:
            return f"Command timed out after {timeout} seconds."

        if result.returncode != 0:
            return f"Exception (exit code {result.returncode}):\n\n{result.stdout}"

        return truncate_output(result.stdout, truncate_at)

    except Exception as e:
        return f"Error executing script: {str(e)}"


python_server.tool(execute)
