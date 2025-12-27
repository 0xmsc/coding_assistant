from __future__ import annotations

from typing import Annotated

from fastmcp import FastMCP

from coding_assistant.mcp.proc import start_process
from coding_assistant.mcp.utils import truncate_output
from coding_assistant.mcp.tasks import TaskManager


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

        The execution supports PEP 723 inline script metadata, allowing you to specify dependencies directly in the code block.

        Example:
        ```python
        # /// script
        # dependencies = ["requests"]
        # ///
        import requests
        print(requests.get("https://github.com").status_code)
        ```

        If the environment variable MCP_SERVER_URL is set, the code will have access to an MCP server. A `fastmcp` dependency will also be present. Use `fastmcp` to access the server and call its tools.

        ```python
        import asyncio
        import os
        from fastmcp import Client

        async def use_mcp():
            mcp_url = os.environ.get("MCP_SERVER_URL")
            async with Client(mcp_url) as client:
                # Example: List available tools
                tools = await client.list_tools()
                print(f"Available tools: {[t.name for t in tools]}")

                # Example: Call tool
                result = await client.call_tool("shell_execute", {"command": "ls"})
                print(result.content[0].text)

        asyncio.run(use_mcp())
        ```

        Use Python to combine results from different tools and build complex pipelines without leaving the Python context. This is preferred for complex logic, data processing, or batch operations involving multiple tool calls.

        ```python
        import asyncio
        import os
        from fastmcp import Client

        async def pipeline():
            mcp_url = os.environ.get("MCP_SERVER_URL")
            async with Client(mcp_url) as client:
                # Example: Pipeline combining shell tools
                # 1. List files using shell
                ls_result = await client.call_tool("shell_execute", {"command": "ls *.md"})
                files = ls_result.content[0].text.strip().split("\n")

                # 2. Process content in Python
                for file in files:
                    # Use cat to read the file content
                    content_result = await client.call_tool("shell_execute", {"command": f"cat {file}"})
                    content = content_result.content[0].text
                    # ... perform complex logic on content ...
                    print(f"Processed {file}")

        asyncio.run(pipeline())
        ```
        """

        code = code.strip()

        args = ["uv", "run", "-q"]
        env = {}
        if mcp_url:
            env["MCP_SERVER_URL"] = mcp_url
            # We use --with fastmcp to ensure Client works out of the box
            args.extend(["--with", "fastmcp"])

        args.append("-")

        try:
            handle = await start_process(
                args=args,
                stdin_input=code,
                env=env,
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
