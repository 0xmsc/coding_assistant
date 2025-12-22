from __future__ import annotations

import json
from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.proc import start_process
from coding_assistant_mcp.utils import truncate_output
from coding_assistant_mcp.tasks import TaskManager

BRIDGE_TEMPLATE = """
import asyncio
import json
from fastmcp import Client

MCP_URL = {mcp_url_repr}

async def get_available_mcp_tools():
    \"\"\"Return a JSON-compatible list of available MCP tools.\"\"\"
    if not MCP_URL:
        return []
    async with Client(MCP_URL) as client:
        tools = await client.list_tools()
        return [
            {{
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }}
            for t in tools
        ]

async def call_mcp_tool(name: str, arguments: dict | None = None):
    \"\"\"Call an MCP tool asynchronously.\"\"\"
    if not MCP_URL:
        raise RuntimeError("MCP URL not configured")
    async with Client(MCP_URL) as client:
        result = await client.call_tool(name, arguments or {{}})
        
        # If the result is a standard CallToolResult with text content, return the text
        if hasattr(result, "content") and result.content:
            text_parts = [c.text for c in result.content if hasattr(c, "text")]
            if text_parts:
                return "\\n".join(text_parts)
        
        return result

def call_mcp_tool_sync(name: str, arguments: dict | None = None):
    \"\"\"Call an MCP tool synchronously.\"\"\"
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(call_mcp_tool(name, arguments))
    
    if loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
    return asyncio.run(call_mcp_tool(name, arguments))

"""

def create_python_server(manager: TaskManager, mcp_url: str | None = None) -> FastMCP:
    python_server = FastMCP("Python")

    @python_server.tool()
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
        
        # Inject the bridge
        bridge_code = BRIDGE_TEMPLATE.format(mcp_url_repr=repr(mcp_url))
        code = bridge_code + "\n" + code

        try:
            # We use --with fastmcp and --with nest-asyncio to ensure the bridge works
            handle = await start_process(
                args=["uv", "run", "-q", "--with", "fastmcp", "--with", "nest-asyncio", "-"],
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

            if handle.exit_code != 0:
                return f"Exception (exit code {handle.exit_code}):\n\n{handle.stdout}"

            return stdout_text

        except Exception as e:
            return f"Error executing script: {str(e)}"

    return python_server
