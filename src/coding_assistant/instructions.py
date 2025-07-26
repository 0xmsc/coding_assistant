import os
from pathlib import Path

from coding_assistant.config import Config

INSTRUCTIONS = """
- Do not initialize a new git repository, unless your client explicitly requests it.
- Do not commit any changes to the git repository, unless your client explicitly requests it.
- When you have made a change to a project, ask the user if you should commit the changes.
- Do not use the 'mcp_filesystem_search_files' tool, use the 'rg' shell command instead.
- Do not install any software on the users computer before asking.
- Do not run any binary using `uvx` or `npx` without asking the user first.
- Almost all of your tasks are related to the codebase you are currently working in. When the user asks a question, be *very* sure before starting a web search that this is what the user wants.
- If you output text, use markdown formatting where appropriate.
""".strip()


def get_instructions(working_directory: Path, config: Config) -> str:
    instructions = INSTRUCTIONS

    local_instructions_path = working_directory / ".coding_assistant" / "instructions.md"
    if local_instructions_path.exists():
        local_instructions = local_instructions_path.read_text().strip()
        instructions = f"{instructions}\n{local_instructions}"

    if config.instructions:
        instructions = f"{instructions}\n{config.instructions.strip()}"

    return instructions
