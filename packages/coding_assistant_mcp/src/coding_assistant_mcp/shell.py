from __future__ import annotations

import asyncio
import re
from typing import Annotated

from prompt_toolkit.shortcuts import create_confirm_session

from fastmcp import FastMCP

shell_server = FastMCP()

_SHELL_CONFIRMATION_PATTERNS: list[str] = []


async def _ask_confirmation(command: str) -> bool:
    prompt_text = f"Execute `{command}`?"
    return await create_confirm_session(prompt_text).prompt_async()


def set_shell_confirmation_patterns(patterns: list[str]) -> None:
    global _SHELL_CONFIRMATION_PATTERNS
    _SHELL_CONFIRMATION_PATTERNS = patterns


async def execute(
    command: Annotated[str, "The shell command to execute. Do not include 'bash -c'."],
    timeout: Annotated[int, "The timeout for the command in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
) -> str:
    """Execute a shell command using bash and return combined stdout/stderr.

    If the command matches any configured confirmation pattern, the user is
    prompted (using prompt_toolkit) to confirm execution (y/yes to proceed).
    """
    command = command.strip()

    matched_pattern: str | None = None
    for pattern in _SHELL_CONFIRMATION_PATTERNS:
        if re.search(pattern, command):
            matched_pattern = pattern
            break

    if matched_pattern:
        confirmed = await _ask_confirmation(command)
        if not confirmed:
            return "Command execution denied."

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        return f"Command timed out after {timeout} seconds."

    if proc.returncode != 0:
        result = f"Returncode: {proc.returncode}\n\n{stdout.decode()}"
    else:
        result = stdout.decode()

    if len(result) > truncate_at:
        note = "\n\n[truncated output due to truncate_at limit]"
        truncated = result[: max(0, truncate_at - len(note))]
        result = truncated + note

    return result


shell_server.tool(execute)
