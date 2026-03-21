from __future__ import annotations

import re
from typing import Any, Protocol

from coding_assistant.tool_results import TextResult


class ToolPolicy(Protocol):
    async def before_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> TextResult | None: ...


class NullToolPolicy:
    async def before_tool_execution(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> TextResult | None:
        return None


async def confirm_tool_if_needed(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    patterns: list[str],
    ui: Any,
) -> TextResult | None:
    for pattern in patterns:
        if re.search(pattern, tool_name):
            question = f"Execute tool `{tool_name}` with arguments `{arguments}`?"
            allowed = await ui.confirm(question)
            if not allowed:
                return TextResult(content="Tool execution denied.")
            break
    return None


async def confirm_shell_if_needed(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    patterns: list[str],
    ui: Any,
) -> TextResult | None:
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
                return TextResult(content="Shell command execution denied.")
            break
    return None


class ConfirmationToolPolicy:
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
    ) -> TextResult | None:
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
