# Global instructions

## General

- Do not install any software before asking the user.
- Do not run any binary using `uvx` or `npx` before asking the user.
- Do not start a web search before asking the user.
- Output text in markdown formatting, where appropriate.
- Before doing any non-trivial changes to the codebase, present a coherent plan to the user.
  The plan should contain at least 3 clarifying questions with answers you are proposing.
  When the user has follow-up questions or comments, update the plan accordingly.
  Do not start implementing the plan until the user has approved the plan.
  When the user answers with an empty message, you can interpret it as approval for the plan.

## Developing

- Do not add obvious, redundant comments to the code.
- Do not document what the code is doing, document why it is doing it.
- Do not document trivial functions or classes.

## Tools

- Call multiple tools in one step where appropriate, to parallelize their execution.

## Repository

- Do not initialize a new git repository before asking the user.
- Do not commit changes before asking the user.
- Do not switch branches before asking the user.

## MCP

- You have access to a custom MCP server (`coding_assistant_mcp`).
- Prefer it where other MCPs provide similar tools.

## Shell

- Use MCP shell tool `shell_execute` to execute shell commands.
- `shell_execute` can run multi-line scripts.
- Example commands: `eza`, `git`, `fd`, `rg`, `gh`, `pwd`.
- Be sure that the command you are running is safe. If you are unsure, ask the user.
- Interactive commands (e.g., `git rebase -i`) are not supported and will block.
- Prefer Shell over Python for simple one-liners.

## Python

- You have access to a Python interpreter via `python_execute`.
- `python_execute` can run multi-line scripts.
- The most common Python libraries are already installed.
- Prefer Python over Shell for complex logic.
- Add comments to your scripts to explain your logic.

## TODO

- Always manage a TODO list while working on your task.
- Use the `todo_*` tools for managing the list.

## Exploring 

- Use `pwd` to determine the project you are working on.
- Use shell tools to explore the codebase, e.g. `fd` or `rg`.

## Editing

- Use `cp` & `mv` to copy/move files. Do not memorize and write contents to copy or move.
- Do not try to use `applypatch` to edit files. Use e.g. `sed` or `edit_file`.
- You can use `sed` to search & replace (e.g. to rename variables).
- Writing full files should be the exception. Try to use `edit_file` to edit existing files.

