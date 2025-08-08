# Coding Assistant

**Coding Assistant** is an advanced Python-based tool that leverages agent-based automation to streamline, support, and automate various coding tasks.

## Key Features

- **Agent Orchestration**: The Orchestrator agent coordinates tasks by delegating work to specialized sub-agents.
- **Resume Functionality**: Resume work from a previous session.
- **Project-Specific Caching**: Conversation history is cached within a `.coding_assistant` directory in your project.
- **Flexible CLI**: Launch, control, and interact with agents from the command line.
- **MCP Server Integration**: Native support for MCP server toolchains (filesystem, fetch/web search, git, Tavily, etc).
- **Sandbox Security**: Landlock-based filesystem sandbox for secure task execution.
- **Shell Command Confirmation**: Prompts for user confirmation before executing potentially harmful shell commands.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <REPO_URL>
    cd coding_assistant
    ```
2.  **Set environment variables:**
    -   Set up environment variables for API keys (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`).
    -   You can use a `.envrc` file with `direnv` or export them manually.

## Quickstart

Run the assistant using the `run.fish` script, which handles everything for you.

**Basic Usage:**
```bash
./run.fish --task "Refactor all function names to snake_case."
```

**Hello World:**
```bash
./run.fish --task "Say 'Hello World'"
```

**Resume Last Session:**
```bash
./run.fish --task "Continue with the previous task." --resume
```

For more options, run:
```bash
./run.fish --help
```

## License

MIT
