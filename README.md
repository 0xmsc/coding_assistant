# Coding Assistant

## Introduction

Coding Assistant is an advanced Python-based project designed to streamline and automate various coding tasks. It employs a multi-agent architecture to handle tasks ranging from planning and research to development and expert review.

## Project Overview

The system comprises specialized agents (orchestrator, planner, developer, researcher, and expert) working together through well-defined tools and protocols. These agents coordinate to break down complex tasks, generate or modify code, perform research, and ensure high-quality outcomes.

## Recent Updates

- Migrated to **litellm** for streamlined LLM interactions.
- Integrated **mcp[cli]** for robust file system operations and tooling.
- Updated environment variable usage for specifying LLM models.
- Enhanced agent architecture with clearer separation of responsibilities.

## Features

- Multi-agent system (orchestrator, planner, developer, researcher, expert) for intelligent task handling
- Automated code generation, refactoring, and review
- Integrated file system operations via MCP
- Modular and extensible architecture
- Strong typing with `mypy` for improved code quality
- Comprehensive test suite with `pytest`
- Custom logging for enhanced debugging and monitoring

## Installation

1. Ensure you have **Python 3.12+** installed on your system.
2. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/coding-assistant.git
   cd coding-assistant
   ```
3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install the package and its dependencies:
   ```bash
   pip install .
   ```

## Configuration

Configure the LLM models via environment variables:

```bash
export CODING_ASSISTANT_MODEL="o4-mini"
export CODING_ASSISTANT_EXPERT_MODEL="<disabled>"
```

Default values:
- `CODING_ASSISTANT_MODEL`: `o4-mini`
- `CODING_ASSISTANT_EXPERT_MODEL`: `<disabled>`

## Usage Guide

After installation, invoke the assistant using the provided CLI:

```bash
coding-assistant --task "Describe your task here"
```

Alternatively, run via the Python module:

```bash
python -m coding_assistant.main --task "Describe your task here"
```

## Project Structure

```
.
├── src/
│   └── coding_assistant/
│       ├── agents/
│       ├── llm/
│       ├── config.py
│       ├── main.py
│       └── tools.py
├── tests/
├── pyproject.toml
└── README.md
```

- `src/coding_assistant/agents/`: Orchestrator, planner, developer, expert, and researcher agent implementations.
- `src/coding_assistant/llm/`: Interface for LLM interactions via litellm.
- `src/coding_assistant/config.py`: Configuration definitions.
- `src/coding_assistant/main.py`: CLI entry point.
- `src/coding_assistant/tools.py`: Utilities for MCP-based filesystem and other servers.

## Technologies Used

- Python 3.12+
- litellm
- mcp[cli]
- pytest
- Git
- direnv (optional)

## Contributing

We welcome contributions to the Coding Assistant project! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests to ensure everything passes: `pytest`
5. Commit your changes: `git commit -am 'Add some feature'`
6. Push to the branch: `git push origin feature/your-feature-name`
7. Open a Pull Request

Please ensure tests are updated as appropriate and adhere to the project's coding standards. For major changes, open an issue first to discuss your proposal.

## Acknowledgements

- Thanks to all contributors who have helped shape Coding Assistant.
- Special thanks to the developers of litellm, mcp, pytest, and other libraries that power this project.