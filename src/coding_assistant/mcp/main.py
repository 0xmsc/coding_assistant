from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.utilities.logging import configure_logging

from coding_assistant.mcp.filesystem import filesystem_server
from coding_assistant.mcp.python import create_python_server
from coding_assistant.mcp.shell import create_shell_server
from coding_assistant.mcp.todo import create_todo_server
from coding_assistant.mcp.tasks import create_task_server, TaskManager
from coding_assistant.mcp.skills import create_skills_server

INSTRUCTIONS = """
## Shell

- Use MCP shell tool `shell_execute` to execute shell commands.
- `shell_execute` can run multi-line scripts.
- Example commands: `eza`, `git`, `fd`, `rg`, `gh`, `pwd`.
- Be sure that the command you are running is safe. If you are unsure, ask the user.
- Interactive commands (e.g., `git rebase -i`) are not supported and will block.

## Python

- You have access to a Python interpreter via `python_execute`.
- `python_execute` can run multi-line scripts.
- Prefer Python over Shell for readability.
- Add comments to your scripts to explain your logic.

## TODO

- Always manage a TODO list while working on your task.
- Use the `todo_*` tools for managing the list.

## Filesystem

- Use filesystem tools to read, write, and edit files.
- Try not to use shell commands for file operations.

## Tasks

- Use tasks tools to monitor and manage background tasks.
""".strip()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Coding Assistant MCP Server")
    parser.add_argument("--mcp-url", help="The URL of the MCP server (passed to Python scripts)")
    parser.add_argument(
        "--skills-directories",
        nargs="*",
        default=[],
        help="Paths to directories containing Agent Skills (with SKILL.md files).",
    )
    args = parser.parse_args()

    configure_logging(level="CRITICAL")

    manager = TaskManager()

    skills_dir = [Path(d) for d in args.skills_directories]
    skills_server, skills_instr = create_skills_server(skills_dir)
    instructions = f"{INSTRUCTIONS}\n\n{skills_instr}"

    mcp = FastMCP("Coding Assistant MCP", instructions=instructions)
    await mcp.import_server(create_todo_server(), prefix="todo")
    await mcp.import_server(create_shell_server(manager), prefix="shell")
    await mcp.import_server(create_python_server(manager, args.mcp_url), prefix="python")
    await mcp.import_server(filesystem_server, prefix="filesystem")
    await mcp.import_server(create_task_server(manager), prefix="tasks")
    await mcp.import_server(skills_server, prefix="skills")
    await mcp.run_async(show_banner=False)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
