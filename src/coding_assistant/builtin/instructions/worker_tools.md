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

## Attachments

- Use `load_file(path)` before reasoning from an uploaded session attachment.
- `load_file` only reads files under the session `attachments/` directory.
- It can load bounded UTF-8 text and supported image attachments into conversation context.

## Tasks

- Use tasks tools to monitor and manage background tasks.
