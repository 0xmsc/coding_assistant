from __future__ import annotations

import asyncio

from fastmcp import FastMCP
from fastmcp.utilities.logging import configure_logging

from coding_assistant_mcp.filesystem import filesystem_server
from coding_assistant_mcp.python import python_server
from coding_assistant_mcp.shell import shell_server
from coding_assistant_mcp.todo import create_todo_server


async def _main() -> None:
    # Set logging to CRITICAL to minimize output from FastMCP
    configure_logging(level="CRITICAL")

    mcp = FastMCP("Coding Assistant MCP", instructions="")
    await mcp.import_server(create_todo_server(), prefix="todo")
    await mcp.import_server(shell_server, prefix="shell")
    await mcp.import_server(python_server, prefix="python")
    await mcp.import_server(filesystem_server, prefix="filesystem")
    await mcp.run_async()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
