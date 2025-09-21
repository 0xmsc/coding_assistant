from __future__ import annotations
import asyncio

from fastmcp import FastMCP

from coding_assistant_mcp.todo import create_todo_server
from coding_assistant_mcp.shell import shell_server


async def _main() -> None:
    mcp = FastMCP("Coding Assistant MCP")
    # Create a new isolated todo server instance
    todo_server = create_todo_server()
    await mcp.import_server(todo_server, prefix="todo")
    await mcp.import_server(shell_server, prefix="shell")
    await mcp.run_async()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
