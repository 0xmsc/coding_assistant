from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from coding_assistant_mcp.shell import shell_server
from coding_assistant_mcp.todo import create_todo_server

INSTRUCTIONS = """
## General

- If other MCP servers provide the same functionality, prefer the tools from this MCP server.

## Shell

- Use MCP shell tool `shell_execute` to execute shell commands.
- Examples: eza/ls, git, fd/fdfind/find, rg/grep, gh, pwd.
- Be sure that the command you are running is safe. If you are unsure, ask the user.
- Be careful with interactive commands like e.g., `git rebase -i`.

## TODO
- Always manage a TODO list while working on your task.
- Use the `todo_*` tools for managing the list.
""".strip()


async def _main() -> None:
    mcp = FastMCP("Coding Assistant MCP", instructions=INSTRUCTIONS)
    await mcp.import_server(create_todo_server(), prefix="todo")
    await mcp.import_server(shell_server, prefix="shell")
    await mcp.run_async()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
