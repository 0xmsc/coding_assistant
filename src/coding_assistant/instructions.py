from __future__ import annotations

import logging
from pathlib import Path

from coding_assistant.tools.mcp import MCPServer

logger = logging.getLogger(__name__)

INSTRUCTIONS = """
# Global instructions

## General

- Do not install any software on the users computer before asking.
- Do not run any binary using `uvx` or `npx` without asking the user first.
- When the user asks a question, be *very* sure before starting a web search that this is what the user wants.
- If you output text, use markdown formatting where appropriate.

## Repository

- Do not initialize a new git repository, unless your client explicitly requests it.
- Do not commit changes without asking the user first.
- When you have made a change to a project, ask the user if you should commit the changes.
- Do not switch branches before asking the user.

## Sub-agents

- Make use of sub-agents to reduce your context size.
- If you expect to read lots of files to gather information, launch a sub-agent.
- When possible, launch multiple sub-agents in parallel to speed up your work.
- It is very important to pass all necessary context to the sub-agent, they do not have access to your conversation history with the client. The only thing they see is the parameters you pass to them.
- You are responsible for the work of the sub-agents. Review it before showing it to the client.
""".strip()

PLANNING_INSTRUCTIONS = """
# Planning mode

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


def get_instructions(
    working_directory: Path,
    plan: bool,
    user_instructions: list[str],
    mcp_servers: list[MCPServer] | None = None,
) -> str:
    sections: list[str] = []

    sections.append(INSTRUCTIONS.strip())

    if plan:
        sections.append(PLANNING_INSTRUCTIONS.strip())

    for path in [
        working_directory / ".coding_assistant" / "instructions.md",
        working_directory / "AGENTS.md",
    ]:
        if not path.exists():
            continue

        content = path.read_text().strip()
        if not content:
            continue

        sections.append(content)

    for server in mcp_servers or []:
        text = getattr(server, "instructions", None)
        if text and text.strip():
            sections.append(f"# MCP `{server.name}` instructions\n\n{text.strip()}")

    if user_instructions:
        sections.append("# User-provided instructions")
        for user_instruction in user_instructions:
            if user_instruction and user_instruction.strip():
                sections.append(user_instruction.strip())

    for section in sections:
        if not section.startswith("# "):
            logger.warning(f"Instruction section {section} does not start with a top-level heading")

    return "\n\n".join(sections)
