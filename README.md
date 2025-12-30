# Coding Assistant

Coding Assistant is a Python-based, agent-orchestrated CLI that helps you automate and streamline coding tasks. It can plan, launch sub-agents, use MCP tools (filesystem, web fetch/search, Context7, Tavily, etc.), run inside a sandbox, and keep resumable history.

## Key Features

- Orchestrator agent that delegates to sub-agents and tools
- Resumable sessions and conversation summaries stored per-project
- Built-in MCP server with shell, Python, filesystem, and TODO tools
- Support for external MCP servers (filesystem, fetch, Context7, Tavily, etc.)
- Landlock-based filesystem sandbox with readable/writable allowlists
- Prompt-toolkit powered TUI with dense and regular output modes
- Shell/tool confirmation patterns to guard dangerous operations
- Chat mode enabled by default for interactive conversations
- Configurable via CLI flags (models, planning mode, instructions, etc.)

## Requirements

- Python 3.12+
- uv (recommended) or pip for running/installing

- Optional: External MCP servers if you want to extend functionality
  - Node.js/npm for `npx` (for NPM-based MCP servers)
  - Network access to fetch packages
- API keys as needed by your chosen model/tooling, e.g.:
  - `OPENAI_API_KEY` (or other LiteLLM-compatible provider keys)
  - Additional keys for external MCP servers (e.g., `TAVILY_API_KEY`)

## Installation

Using uv (recommended):

```bash
uv tool install coding-assistant-cli
```

## Quickstart

The easiest way to run is with the uv command:

```bash
coding-assistant \
  --model "openrouter/anthropic/claude-3.5-sonnet" \
  --task "Refactor all function names to snake_case."
```

Resume the last session:

```bash
coding-assistant \
  --model "openrouter/anthropic/claude-3.5-sonnet" \
  --resume
```

Show available options:

```bash
coding-assistant --help
```

### Advanced Examples

You can invoke the CLI with additional MCP servers (stdio or SSE/remote). The built-in `coding_assistant.mcp` is included by default.

```bash
coding-assistant \
  --model "openrouter/openai/gpt-4o-mini" \
  --task "Say 'Hello World'" \
  --mcp-servers \
    '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]}' \
    '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}'
```

Notes:
- Model names are routed via LiteLLM. Use any provider/model you have keys for; set the corresponding API key env vars.
- The `--mcp-servers` values are JSON strings; arguments support variable substitution for `{home_directory}` and `{working_directory}`.

## Usage Highlights

- `--model` Select model for the orchestrator agent (required).
- `--expert-model` Select model for expert tasks (defaults to the same value as `--model`).
- `--tool-confirmation-patterns` / `--shell-confirmation-patterns` Regex patterns to require user confirmation before tool/shell execution.
- `--chat-mode` / `--no-chat-mode` Enable/disable open-ended chat mode (default: **enabled**).
- `--resume` / `--resume-file` Resume from the latest/specific orchestrator history in `.coding_assistant/history/`.
- `--instructions` Provide extra instructions that are composed with defaults. Can be repeated.
- `--compact-conversation-at-tokens` Number of tokens after which conversation should be shortened (default: 200,000).
- `--trace` / `--no-trace` Enable/disable tracing of model requests/responses.
- `--sandbox` / `--no-sandbox` Enable/disable Landlock-based sandboxing (default: **enabled**).
- `--wait-for-debugger` Wait for a debugger (debugpy) to attach on port 1234.

Run `coding-assistant --help` to see all options.

## MCP Servers

Pass MCP servers with repeated `--mcp-servers` flags as JSON strings. Support is provided for both stdio-based servers and remote servers via SSE (Server-Sent Events).

### Stdio Server

```json
{
  "name": "filesystem",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]
}
```

### Remote Server (SSE)

```json
{
  "name": "remote-mcp",
  "url": "http://localhost:8000/sse"
}
```

### Built-in: Coding Assistant MCP

This repository includes a built-in MCP server (package `coding_assistant.mcp`) that is started automatically by default. It provides:

- **shell**: `shell_execute` — Execute shell commands with timeout and output truncation
- **python**: `python_execute` — Execute Python code with timeout and output truncation
- **filesystem**: `filesystem_write_file`, `filesystem_edit_file` — Write new files or apply targeted edits
- **todo**: `todo_add`, `todo_list_todos`, `todo_complete` — Simple in-memory TODO list management

When connected, tools are exposed to the agent as fully-qualified names (e.g., `shell_execute`).


### External MCP Servers (Optional)

You can add external MCP servers to extend functionality. Examples include:

- **filesystem**: `@modelcontextprotocol/server-filesystem` (NPM) — Additional filesystem operations
- **fetch**: `mcp-server-fetch` (uvx) — Web fetching capabilities
- **context7**: `@upstash/context7-mcp` (NPM) — Context management
- **tavily**: `tavily-mcp` (needs `TAVILY_API_KEY`) — Web search

To use these, add them via `--mcp-servers` flags as shown in the examples above.

You can print all discovered tools from running MCP servers:

```bash
coding-assistant --print-mcp-tools --mcp-servers '...'
```

## Sandbox

When enabled (default), the assistant applies Landlock restrictions. By default it adds:
- Readable directories: your active virtual environment and any passed via `--readable-sandbox-directories`.
- Writable directories: the current working directory and any passed via `--writable-sandbox-directories`.

Use these flags to widen access if needed when working across multiple directories or mounts.

Example:
```bash
--readable-sandbox-directories /mnt/wsl ~/.ssh ~/.rustup \
--writable-sandbox-directories "$project_dir" /tmp /dev/shm ~/.cache/coding_assistant
```

## History and Resume

- Conversation history and summaries are kept under `.coding_assistant/` in your project.
- Use `--resume` to continue from the most recent session, or `--resume-file` to select a specific file.
- The assistant automatically trims old history once it's saved.

## Shell command execution behavior

The built-in MCP tools `shell_execute` and `python_execute`:
- Support multi-line scripts
- Merge stderr into stdout and return plain text (no JSON envelope)
- Prefix output with `Exit code: N` only when the command/code exits non-zero
- Support `truncate_at` parameter to limit combined output size and append a note when truncation occurs
- Support `timeout` parameter (default: 30 seconds)
- Interactive commands (e.g., `git rebase -i`) are not supported and will block

## Development

- Run tests:

  ```bash
  just test
  ```

  Run integration tests:

  ```bash
  just test-integration
  ```

  or manually:

  ```bash
  uv run pytest -n auto -m "not slow"
  ```

- Run linting/formatting/type-checking:

  ```bash
  just lint
  ```

- Handy `just` recipes: `test`, `lint` (see `justfile`).

## Skills

Coding Assistant includes a skills system that provides specialized workflows, tool integrations, and domain expertise for common tasks. Skills transform the general-purpose agent into a specialized one by bundling instructions, scripts, references, and assets.

### Available Skills

The following skills are available:

- **planning** - Guidelines for iteratively planning tasks and changes before implementation. Use when the user requests a non-trivial task or when you need to align on a complex implementation strategy.

- **code-review** - Provides a structured workflow for planning and executing code reviews like a senior engineer, including checklists for quality, security, and maintainability. Use when asked to review code, PRs, or plan a code review task.

- **create-skill** - Guide for creating new Agent Skills. Use this skill when you need to create a new skill to extend the agent's capabilities with specialized knowledge, workflows, or tool integrations.

- **developing** - General principles for exploring, developing, editing, and refactoring code. Use for codebase analysis, implementation tasks, and code quality improvements.

### How Skills Work

Skills use a progressive disclosure system:

1. **Metadata** (name + description) - Always loaded, ~100 words
2. **SKILL.md body** - Loaded when skill triggers, <500 lines
3. **Bundled resources** - Loaded as needed (scripts, references, assets)

When a skill's context matches the current task, it loads automatically, providing specialized instructions and tools.

### Skill Structure

Skills are organized as directories containing:

```
skill-name/
├── SKILL.md          # Required: instructions + metadata
├── scripts/          # Optional: executable code
├── references/       # Optional: documentation
└── assets/           # Optional: templates/resources
```

## License

MIT
