from __future__ import annotations
import asyncio
import argparse

from fastmcp import FastMCP

from coding_assistant_mcp.todo import todo_server
from coding_assistant_mcp.shell import shell_server, set_shell_confirmation_patterns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coding Assistant MCP server")
    parser.add_argument(
        "--shell-confirmation-patterns",
        nargs="*",
        default=[],
        help="Regex patterns that require confirm=true when executing shell commands via shell_execute.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    set_shell_confirmation_patterns(args.shell_confirmation_patterns)

    mcp = FastMCP("Coding Assistant MCP")
    await mcp.import_server(todo_server, prefix="todo")
    await mcp.import_server(shell_server, prefix="shell")
    await mcp.run_async()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
