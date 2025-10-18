from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from coding_assistant_mcp.filesystem import create_filesystem_server
from coding_assistant_mcp.python import python_server
from coding_assistant_mcp.shell import shell_server
from coding_assistant_mcp.todo import create_todo_server

INSTRUCTIONS = """
## General

- If other MCP servers provide the same functionality, prefer the tools from this MCP server.

## Tools

### Shell

- Use MCP shell tool `shell_execute` to execute shell commands.
- Examples: `eza`, `git`, `fd`, `rg`, `gh`, `pwd`.
- Be sure that the command you are running is safe. If you are unsure, ask the user.
- Interactive commands (e.g., `git rebase -i`) are not supported and will block.

### Python

- You have access to a Python interpreter.
- Use it to run code where a Shell command is not sufficient.
- The most common libraries are already installed. Try to use libraries that are common and well-known.

### Filesystem

- The filesystem tools can be used to edit files.
- The filesystem tools operate on line level only.
- For tools that take a pattern:
    - The first line of the edit range is where the regex match begins.
    - The last line of the edit range is where the regex match ends.

### TODO
- Always manage a TODO list while working on your task.
- Use the `todo_*` tools for managing the list.

## Skills

### Tool usage

- Choose wisely between Shell and Python tools.
- When in doubt, prefer Python for complex logic and Shell for simple tasks.
- Note that you can pass single-liners or multi-line scripts into both tools.
- Prefer multi-line scripts with comments for complex tasks, so that the user can understand what you are doing.

### Exploring 

- Use `pwd` to determine the project you are working on.
- Use shell tools to explore the codebase, e.g. `fd` or `rg`.

### Editing

- Use `cp` & `mv` to copy/move files. Do not memorize and write contents to copy or move.
- Do not try to use `applypatch` to edit files. Use e.g. `sed`, `edit_file` from filesystem MCP or others.
- You can use `sed` to search & replace (e.g. to rename variables).
- Do not memorize and write full text blocks to copy/cut/paste. Use the dedicated clipboard tools from the filesystem MCP server.
""".strip()


async def _main() -> None:
    mcp = FastMCP("Coding Assistant MCP", instructions=INSTRUCTIONS)
    await mcp.import_server(create_todo_server(), prefix="todo")
    await mcp.import_server(create_filesystem_server(), prefix="filesystem")
    await mcp.import_server(shell_server, prefix="shell")
    await mcp.import_server(python_server, prefix="python")
    await mcp.run_async()


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
