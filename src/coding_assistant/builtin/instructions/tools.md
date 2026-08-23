# Local tools

## Shell

- Use `shell_execute` to execute shell commands.
- `shell_execute` can run multi-line scripts.
- Example commands: `eza`, `git`, `fd`, `rg`, `gh`, `pwd`.
- Create a temporary directory (via `mktemp -d`) if you want to write temporary files.
- Be sure that the command you are running is safe. If you are unsure, ask the user.
- Interactive commands (e.g., `git rebase -i`) are not supported and will block.

## Python

- Use `python_execute` to run Python code without shell quoting.
- `python_execute` supports multi-line scripts and PEP 723 inline dependencies.

## Filesystem

- Use filesystem tools to read, write, and edit files.
- Try not to use shell commands for file operations.

## Tasks

- Use tasks tools to monitor and manage background tasks.
