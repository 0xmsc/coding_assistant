from __future__ import annotations

import argparse
import asyncio

from fastmcp import FastMCP

from coding_assistant_mcp.filesystem import filesystem_server
from coding_assistant_mcp.python import create_python_server
from coding_assistant_mcp.shell import create_shell_server
from coding_assistant_mcp.todo import create_todo_server
from coding_assistant_mcp.tasks import create_task_server, TaskManager


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Coding Assistant MCP Server")
    parser.add_argument(
        "--transport", default="streamable-http", choices=["stdio", "streamable-http"], help="Transport to use"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP transport")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transport")
    args = parser.parse_args()

    # Note: configure_logging removed as per request

    manager = TaskManager()

    mcp_url = None
    if args.transport == "streamable-http":
        url_host = args.host if args.host != "0.0.0.0" else "localhost"
        mcp_url = f"http://{url_host}:{args.port}/mcp"

    mcp = FastMCP("Coding Assistant MCP", instructions="")

    await mcp.import_server(create_todo_server(), prefix="todo")
    await mcp.import_server(create_shell_server(manager), prefix="shell")
    await mcp.import_server(create_python_server(manager, mcp_url), prefix="python")
    await mcp.import_server(filesystem_server, prefix="filesystem")
    await mcp.import_server(create_task_server(manager), prefix="tasks")

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
