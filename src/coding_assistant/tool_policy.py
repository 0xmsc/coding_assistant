from __future__ import annotations

import re
from typing import Any, Protocol


class ToolPolicy(Protocol):
    """Hook for approving, blocking, or short-circuiting tool execution."""

    async def before_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Return a tool result to skip execution, or `None` to allow it."""
        ...


class NullToolPolicy:
    """Policy that always allows the tool call to proceed."""

    async def before_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Allow the tool call without modification."""
        return None


async def confirm_tool_if_needed(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    patterns: list[str],
    ui: Any,
) -> str | None:
    """Ask the user before running tools whose names match a configured pattern."""
    for pattern in patterns:
        if re.search(pattern, tool_name):
            question = f"Execute tool `{tool_name}` with arguments `{arguments}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return "Tool execution denied."
            break
    return None


async def confirm_shell_if_needed(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    patterns: list[str],
    ui: Any,
) -> str | None:
    """Ask the user before running shell commands that match a configured pattern."""
    if tool_name != "shell_execute":
        return None

    command = arguments.get("command")
    if not isinstance(command, str):
        return None

    for pattern in patterns:
        if re.search(pattern, command):
            question = f"Execute shell command `{command}` for tool `{tool_name}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return "Shell command execution denied."
            break
    return None


class ConfirmationToolPolicy:
    """Policy that prompts before matching tool or shell executions."""

    def __init__(
        self,
        *,
        ui: Any,
        tool_confirmation_patterns: list[str] | None = None,
        shell_confirmation_patterns: list[str] | None = None,
    ) -> None:
        self._ui = ui
        self._tool_patterns = tool_confirmation_patterns or []
        self._shell_patterns = shell_confirmation_patterns or []

    async def before_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str | None:
        """Apply the configured confirmation checks before a tool runs."""
        if result := await confirm_tool_if_needed(
            tool_name=tool_name,
            arguments=arguments,
            patterns=self._tool_patterns,
            ui=self._ui,
        ):
            return result

        if result := await confirm_shell_if_needed(
            tool_name=tool_name,
            arguments=arguments,
            patterns=self._shell_patterns,
            ui=self._ui,
        ):
            return result

        return None
