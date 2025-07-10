# Coding Assistant

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/coding-assistant)](https://pypi.org/project/coding-assistant)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/coding-assistant/ci.yml?branch=main)](https://github.com/yourusername/coding-assistant/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Coding Assistant is an experimental, Python-based multi-agent system designed to automate and streamline coding workflows. It leverages specialized agents—Orchestrator, Researcher, and Developer—to collaboratively tackle complex tasks, from research and planning through code generation and review.

## Features

- Modular, multi-agent architecture for scalable task handling
- Automated code generation and refactoring
- Research and knowledge retrieval via a dedicated Researcher agent
- Integrated file system and Git operations via MCP for robust tooling
- Strong static typing support and structured configuration

## Technologies & Libraries

- Python 3.12+
- **litellm**: Simplified LLM interaction, supporting multiple providers
- **mcp[cli]**: Filesystem and tooling operations
- **rich**: Rich text and console formatting
- OpenTelemetry: Tracing and telemetry integration

## Recent Updates

- History Trimming: Introduced a MAX_HISTORY constant and implemented trim_history logic to prevent unbounded growth of conversation history by capping stored messages.
- Error Handling Improvements: Replaced internal assert statements with explicit exception raising to provide clearer error messages and improve runtime stability.
- Typo Fix in LLM Model: Corrected a typo in the LLM model usage by renaming mesages to messages for proper parameter handling.
- Import Cleanup: Removed unused imports across modules to streamline code and reduce unnecessary dependencies.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/coding-assistant.git
   cd coding-assistant
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the package:
   ```bash
   pip install .
   ```

## Configuration

The assistant uses environment variables to configure its models and behavior. Create a `.env` or `.envrc` file in the project root with the following (optional):

```bash
CODING_ASSISTANT_MODEL="o4-mini"
CODING_ASSISTANT_EXPERT_MODEL="o3"
```

Default values:

- `CODING_ASSISTANT_MODEL`: `o4-mini`
- `CODING_ASSISTANT_EXPERT_MODEL`: `o3`

## Usage

Invoke the assistant via CLI:

```bash
coding-assistant --task "Describe your task here"
```

Or run as a module:

```bash
python -m coding_assistant.main --task "Describe your task here"
```

## Project Structure

```
.
├── src/coding_assistant/
│   ├── agents/        # Orchestrator, Researcher, Developer implementations
│   ├── llm/           # LLM integration via litellm
│   ├── config.py      # Configuration settings
│   ├── main.py        # CLI entry point
│   └── tools.py       # MCP-based utilities
├── pyproject.toml     # Project metadata and dependencies
├── README.md          # This file
└── .envrc             # direnv configuration (if used)
```

## Contributing

We welcome contributions! To contribute:

1. Fork this repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them.
4. Ensure your code passes any linters or style checks you use.
5. Commit and push:
   ```bash
   git commit -m "feat: add new feature"
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request describing your changes.

Please follow the project's coding standards and update any relevant documentation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
