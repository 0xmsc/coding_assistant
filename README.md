# Coding Assistant

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyPI Version](https://img.shields.io/pypi/v/coding-assistant)](https://pypi.org/project/coding-assistant)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/coding-assistant/ci.yml?branch=main)](https://github.com/yourusername/coding-assistant/actions)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Coding Assistant is an experimental, Python-based multi-agent system designed to automate and streamline coding workflows. It leverages specialized agents—Orchestrator, Planner, Developer, Researcher, and Expert—to collaboratively tackle complex tasks, from planning and research to code generation and review.

## Features

- Multi-agent architecture for scalable, modular task handling
- Automated code generation, refactoring, and expert review
- Research and knowledge retrieval with a dedicated Researcher agent
- Integrated file system operations via MCP for robust tooling
- Strong static typing with `mypy` and comprehensive testing with `pytest`
- Customizable logging and configuration for flexible workflows

## Technologies & Libraries

- Python 3.12+
- **litellm**: Simplified LLM interaction
- **mcp[cli]**: Filesystem operations and tooling
- **pytest**: Testing framework
- **mypy**: Static type checking
- **pre-commit**: Git hooks management
- **black**, **isort**: Code formatting and import sorting

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
3. Install the package and development dependencies:
   ```bash
   pip install .
   pip install --upgrade ".[dev]"
   ```
4. (Optional) Set up Git hooks:
   ```bash
   pre-commit install
   ```

## Configuration

The assistant uses environment variables to configure LLM models and behavior. Create a `.env` or `.envrc` file in the project root with the following:

```bash
CODING_ASSISTANT_MODEL="o4-mini"
CODING_ASSISTANT_EXPERT_MODEL="<disabled>"
```

Default values:

- `CODING_ASSISTANT_MODEL`: `o4-mini`
- `CODING_ASSISTANT_EXPERT_MODEL`: `<disabled>`

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
│   ├── agents/        # Orchestrator, Planner, Developer, Researcher, Expert
│   ├── llm/           # LLM integration via litellm
│   ├── config.py      # Configuration settings
│   ├── main.py        # CLI entry point
│   └── tools.py       # MCP-based utilities
├── tests/             # Test suite (pytest)
├── pyproject.toml     # Project metadata and dependencies
├── README.md          # This file
└── .envrc             # direnv configuration
```

## Contributing

We welcome contributions! To contribute:

1. Fork this repository
2. Create your feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and add tests
4. Run tests and linters:
   ```bash
   pytest
   pre-commit run --all-files
   ```
5. Commit and push:
   ```bash
   git commit -am "feat: add new feature"
   git push origin feature/your-feature-name
   ```
6. Open a Pull Request

Please follow the project's coding standards and update tests as needed.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
