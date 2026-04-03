# Local tools

## Shell

- Use `shell_execute` to execute shell commands.
- `shell_execute` can run multi-line scripts.
- Example commands: `eza`, `git`, `fd`, `rg`, `gh`, `pwd`.
- Create a temporary directory (via `mktemp -d`) if you want to write temporary files.
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

## Remotes

- Use `remotes_discover()` to find other locally advertised `coding-assistant` instances on this machine.
- Use `remote_connect(endpoint=...)` with the remote endpoint printed when `coding-assistant` starts, then use the returned local `remote_id` with the other remote tools.
- Use `remote_prompt(remote_id=..., prompt=...)` only when the remote is idle. If it is busy, wait for it to finish or use `remote_cancel(...)` before prompting again.

## MCP

- MCP servers must be configured via CLI `--mcp-servers` to be available.
- Use `mcp_start(server="...")` to start an MCP server and connect to it.
- Use `mcp_stop(server="...")` to stop a running MCP server.
- Use `mcp_list_tools(server="...")` to see what tools a server provides.
- Use `mcp_call(server="...", tool="...", arguments={...})` to call a tool on a running MCP server.
- MCP servers are lazily loaded - you control when to start/stop them.
