# Coding Assistant

Coding Assistant is a Python-based CLI and embeddable library for coding workflows. It combines a boundary-based agent loop, built-in local tools, and optional external MCP servers.

## Key Features

- Boundary-based core API with `run_agent_until_boundary(...)`.
- Simple `run_agent(history=...)` convenience wrapper.
- Caller-owned history with `compact_history(...)`.
- Built-in local shell, Python, filesystem, TODO, and background-task tools.
- Support for external MCP servers over stdio or SSE.
- Prompt-toolkit powered interactive CLI.
- Optional externally supplied skill directories.
- No built-in sandboxing; run it inside an external sandbox if you need isolation.

## Requirements

- Python 3.12+.
- `uv` is recommended for installation and running.
- API keys for your chosen OpenAI-compatible provider, for example `OPENAI_API_KEY` or `OPENROUTER_API_KEY`.
- Optional dependencies for external MCP servers, such as Node.js/npm for NPM-based servers.

## Installation

Using `uv`:

```bash
uv tool install coding-assistant-cli
```

For local development:

```bash
uv sync
```

## Quickstart

Run a one-shot task:

```bash
coding-assistant \
  --model "openrouter/anthropic/claude-3.5-sonnet" \
  --task "Refactor all function names to snake_case."
```

Start an interactive session:

```bash
coding-assistant --model "openai/gpt-5-mini"
```

Show available options:

```bash
coding-assistant --help
```

## Embedding

The simplest Python surface is `run_agent(...)`. You pass in history, model, and executable tools; the function streams assistant output through `on_content` and returns the updated transcript after executing any requested tools.

```python
import asyncio
from typing import Any

from coding_assistant import run_agent
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import SystemMessage, Tool, UserMessage


class LookupDocsTool(Tool):
    def name(self) -> str:
        return "lookup_docs"

    def description(self) -> str:
        return "Look up project documentation."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def execute(self, parameters: dict[str, Any]) -> str:
        return f"Documentation for: {parameters['query']}"


async def main() -> None:
    history = [
        SystemMessage(content=build_system_prompt(instructions="You are a helpful coding agent.")),
        UserMessage(content="Say hello in one sentence."),
    ]

    history = await run_agent(
        history=history,
        model="openai/gpt-5-mini",
        tools=[LookupDocsTool()],
        on_content=lambda chunk: print(chunk, end="", flush=True),
    )


asyncio.run(main())
```

If you want explicit control boundaries, use `run_agent_until_boundary(...)` and `execute_tool_calls(...)`. The lower-level API returns `AwaitingUser` or `AwaitingTools`, so callers can own the surrounding loop while still reusing tool execution.

## CLI Highlights

- `--model` selects the model to use. Required.
- `--task` seeds a one-shot or non-interactive run.
- `--instructions` appends custom instructions.
- `--mcp-servers` configures external MCP servers as JSON strings.
- `--print-mcp-tools` prints the discovered MCP tools and exits.
- `--trace` writes model request and response traces.
- `--wait-for-debugger` waits for a debugger on port `1234`.
- `--ask-user` controls whether `--task` runs prompt for follow-up input.
- `--skills-directories` loads optional skill directories.

Interactive prompting is enabled by default when no `--task` is provided.

The interactive CLI also supports:

- `/exit`
- `/help`
- `/compact`
- `/image <path-or-url>`

## MCP Servers

Pass MCP servers with repeated `--mcp-servers` flags as JSON strings. Both stdio-based servers and remote SSE servers are supported.

Example stdio server:

```json
{
  "name": "filesystem",
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]
}
```

Example SSE server:

```json
{
  "name": "remote-mcp",
  "url": "http://localhost:8000/sse"
}
```

Arguments support variable substitution for `{home_directory}` and `{working_directory}`.

Example:

```bash
coding-assistant \
  --model "openrouter/openai/gpt-4o-mini" \
  --task "Say Hello World" \
  --mcp-servers \
    '{"name": "filesystem", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "{home_directory}"]}' \
    '{"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}'
```

## Built-In Local Tools

The default CLI includes these local tools:

- `shell_execute` for running shell commands.
- `python_execute` for running Python snippets.
- `filesystem_write_file` and `filesystem_edit_file` for targeted file changes.
- `todo_add`, `todo_list_todos`, and `todo_complete` for TODO tracking.
- `tasks_list_tasks`, `tasks_get_status`, `tasks_get_output`, `tasks_kill_task`, and `tasks_remove_task` for background task management.

The core runtime also adds internal helper tools such as `compact_conversation` and `redirect_tool_call`.

## Skills

Coding Assistant no longer ships bundled skills.

If you want skills, point the CLI at one or more directories with `--skills-directories`. Each skill directory should contain child directories with a `SKILL.md` file:

```text
skills-root/
├── my-skill/
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
└── another-skill/
    └── SKILL.md
```

When skill directories are configured, the agent gets:

- `skills_list_resources`
- `skills_read`

Those tools expose only the files inside the configured skill directories.

## External Sandboxing

Built-in sandboxing has been removed from this project. If you want filesystem isolation, run the assistant inside an external sandbox such as `bubblewrap`.

See [docs/sandboxing.md](docs/sandboxing.md) for a minimal `bubblewrap` example.

## Shell And Python Tool Behavior

The built-in `shell_execute` and `python_execute` tools:

- support multi-line scripts,
- merge stderr into stdout,
- prefix output with an exit-code header only for non-zero exits,
- support `truncate_at` to limit output size,
- support `timeout` with a default of 30 seconds,
- can hand long-running work off to the background task manager.

Interactive terminal programs such as `git rebase -i` are not supported.

## Development

Run tests:

```bash
just test
```

Run linting, formatting, and type checks:

```bash
just lint
```

There is also an integration test target:

```bash
just test-integration
```

## License

MIT
