from __future__ import annotations

import argparse
import asyncio

from fastmcp import FastMCP
from fastmcp.utilities.logging import configure_logging

from coding_assistant_mcp.filesystem import filesystem_server
from coding_assistant_mcp.python import create_python_server
from coding_assistant_mcp.shell import create_shell_server
from coding_assistant_mcp.todo import create_todo_server
from coding_assistant_mcp.tasks import create_task_server, TaskManager


def create_mcp_server(
    transport: str = "streamable-http",
    host: str = "0.0.0.0",
    port: int = 8000,
) -> tuple[FastMCP, TaskManager, str | None]:
    manager = TaskManager()

    mcp_url = None
    if transport in ["streamable-http", "sse"]:
        url_host = host if host != "0.0.0.0" else "localhost"
        mcp_url = f"http://{url_host}:{port}/mcp"

    mcp = FastMCP("Coding Assistant MCP", instructions="")
    setattr(mcp, "manager", manager)  # For testing accessibility

    # We use asyncio.run or similar eventually, but here we just need to return the server
    # Note: FastMCP.import_server is sync in recent versions but we should be careful
    # if it's async in this specific environment. In main.py it was awaited.
    return mcp, manager, mcp_url


async def setup_server(mcp: FastMCP, manager: TaskManager, mcp_url: str | None) -> FastMCP:
    await mcp.import_server(create_todo_server(), prefix="todo")
    await mcp.import_server(create_shell_server(manager), prefix="shell")
    await mcp.import_server(create_python_server(manager, mcp_url), prefix="python")
    await mcp.import_server(filesystem_server, prefix="filesystem")
    await mcp.import_server(create_task_server(manager), prefix="tasks")
    return mcp


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Coding Assistant MCP Server")
    parser.add_argument(
        "--transport", default="streamable-http", choices=["stdio", "streamable-http", "sse"], help="Transport to use"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP transport")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    args = parser.parse_args()

    configure_logging(level="CRITICAL")

    mcp, manager, mcp_url = create_mcp_server(args.transport, args.host, args.port)
    await setup_server(mcp, manager, mcp_url)

    await mcp.run_async(
        transport=args.transport,
        host=args.host,
        port=args.port,
        show_banner=False,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
