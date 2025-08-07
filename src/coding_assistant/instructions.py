import os
from pathlib import Path
from typing import List

from coding_assistant.config import Config

INSTRUCTIONS = """
- Do not initialize a new git repository, unless your client explicitly requests it.
- Do not commit any changes to the git repository, unless your client explicitly requests it.
- Do not run anything interactively, e.g. `git rebase -i`.
- When you have made a change to a project, ask the user if you should commit the changes.
- Do not use the 'mcp_filesystem_search_files' tool, use the 'rg' shell command instead.
- Do not install any software on the users computer before asking.
- Do not run any binary using `uvx` or `npx` without asking the user first.
- Almost all of your tasks are related to the codebase you are currently working in. When the user asks a question, be *very* sure before starting a web search that this is what the user wants.
- If you output text, use markdown formatting where appropriate.
""".strip()

PLANNING_INSTRUCTIONS = """
- You are in planning mode.
    - Create a plan for the task at hand in close collaboration with the client.
    - Do not implement the plan.
    - Do not make any filesystem changes, except for saving the plan.
    - Present pros and cons of different approaches to the client.
    - Ask the client for feedback on the plan.
    - Planning might take multiple iterations.
    - The default directory to save the plan to is .coding_assistant/plans in the current working directory.
    - Come up with a sensible filename for the plan.
    - Each plan should include a section with a detailed description of the problem and the implementation.
    - It should be clear why this implementation has been chosen over others.
    - Each plan should include a list of tasks that need to be completed to implement the plan.
    - Use a markdown task list for the tasks, such that the tasks can be checked off.
""".strip()


def get_instructions(working_directory: Path, plan: bool, user_instructions: List[str]) -> str:
    instructions = INSTRUCTIONS.strip()

    if plan:
        instructions = f"{instructions}\n{PLANNING_INSTRUCTIONS.strip()}"

    local_instructions_path = working_directory / ".coding_assistant" / "instructions.md"
    if local_instructions_path.exists():
        instructions = f"{instructions}\n{local_instructions_path.read_text().strip()}"

    for instruction in user_instructions:
        instructions = f"{instructions}\n{instruction.strip()}"

    return instructions
