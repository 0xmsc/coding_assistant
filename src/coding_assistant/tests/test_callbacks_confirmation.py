import pytest

from coding_assistant.callbacks import (
    confirm_tool_if_needed,
    confirm_shell_if_needed,
    ConfirmationToolCallbacks,
)
from coding_assistant.tool_results import TextResult
from unittest.mock import AsyncMock, Mock


def make_ui_mock(
    *,
    ask_sequence: list[tuple[str, str]] | None = None,
    confirm_sequence: list[tuple[str, bool]] | None = None,
) -> Mock:
    ui = Mock()

    ask_seq = list(ask_sequence) if ask_sequence is not None else None
    confirm_seq = list(confirm_sequence) if confirm_sequence is not None else None

    async def _ask(prompt_text: str, default: str | None = None) -> str:
        assert ask_seq is not None, "UI.ask was called but no ask_sequence was provided"
        assert len(ask_seq) > 0, "UI.ask was called more times than expected"
        expected_prompt, value = ask_seq.pop(0)
        assert prompt_text == expected_prompt, f"Unexpected ask prompt. Expected: {expected_prompt}, got: {prompt_text}"
        return value

    async def _confirm(prompt_text: str) -> bool:
        assert confirm_seq is not None, "UI.confirm was called but no confirm_sequence was provided"
        assert len(confirm_seq) > 0, "UI.confirm was called more times than expected"
        expected_prompt, value = confirm_seq.pop(0)
        assert prompt_text == expected_prompt, (
            f"Unexpected confirm prompt. Expected: {expected_prompt}, got: {prompt_text}"
        )
        return bool(value)

    async def _prompt(words: list[str] | None = None) -> str:
        return await _ask("> ", None)

    ui.ask = AsyncMock(side_effect=_ask)
    ui.confirm = AsyncMock(side_effect=_confirm)
    ui.prompt = AsyncMock(side_effect=_prompt)
    return ui


@pytest.mark.asyncio
async def test_confirm_tool_if_needed_denied_and_allowed() -> None:
    tool_name = "dangerous_tool"
    arguments = {"path": "/tmp/file.txt"}
    prompt = f"Execute tool `{tool_name}` with arguments `{arguments}`?"
    ui = make_ui_mock(confirm_sequence=[(prompt, False), (prompt, True)])

    res = await confirm_tool_if_needed(
        tool_name=tool_name,
        arguments=arguments,
        patterns=[r"dangerous_"],
        ui=ui,
    )
    assert isinstance(res, TextResult)
    assert res.content == "Tool execution denied."

    res2 = await confirm_tool_if_needed(
        tool_name=tool_name,
        arguments=arguments,
        patterns=[r"dangerous_"],
        ui=ui,
    )
    assert res2 is None


@pytest.mark.asyncio
async def test_confirm_tool_if_needed_no_match_no_prompt() -> None:
    ui = make_ui_mock()  # no confirm sequence expected
    res = await confirm_tool_if_needed(
        tool_name="safe_tool",
        arguments={"x": 1},
        patterns=[r"dangerous_"],
        ui=ui,
    )
    assert res is None


@pytest.mark.asyncio
async def test_confirm_shell_if_needed_denied_and_allowed() -> None:
    tool_name = "shell_execute"
    command = "rm -rf /tmp"
    args = {"command": command}
    prompt = f"Execute shell command `{command}` for tool `{tool_name}`?"
    ui = make_ui_mock(confirm_sequence=[(prompt, False), (prompt, True)])

    # Denied
    res = await confirm_shell_if_needed(
        tool_name=tool_name,
        arguments=args,
        patterns=[r"rm -rf"],
        ui=ui,
    )
    assert isinstance(res, TextResult)
    assert res.content == "Shell command execution denied."

    res2 = await confirm_shell_if_needed(
        tool_name=tool_name,
        arguments=args,
        patterns=[r"rm -rf"],
        ui=ui,
    )
    assert res2 is None


@pytest.mark.asyncio
async def test_confirm_shell_if_needed_ignores_other_tools_and_bad_command() -> None:
    ui = make_ui_mock()
    res = await confirm_shell_if_needed(
        tool_name="some_other_tool",
        arguments={"command": "rm -rf /tmp"},
        patterns=[r"rm -rf"],
        ui=ui,
    )
    assert res is None

    res2 = await confirm_shell_if_needed(
        tool_name="shell_execute",
        arguments={"command": ["echo", "hi"]},
        patterns=[r"echo"],
        ui=ui,
    )
    assert res2 is None


@pytest.mark.asyncio
async def test_confirmation_tool_callbacks_tool_pattern() -> None:
    callbacks = ConfirmationToolCallbacks(
        tool_confirmation_patterns=[r"^my_tool$"],
        shell_confirmation_patterns=[r"will_not_match"],
    )
    tool_name = "my_tool"
    args = {"a": 1}
    prompt = f"Execute tool `{tool_name}` with arguments `{args}`?"
    ui = make_ui_mock(confirm_sequence=[(prompt, False), (prompt, True)])

    # Denied
    res = await callbacks.before_tool_execution(
        context_name="Agent",
        tool_call_id="1",
        tool_name=tool_name,
        arguments=args,
        ui=ui,
    )
    assert isinstance(res, TextResult)
    assert res.content == "Tool execution denied."

    res2 = await callbacks.before_tool_execution(
        context_name="Agent",
        tool_call_id="2",
        tool_name=tool_name,
        arguments=args,
        ui=ui,
    )
    assert res2 is None


@pytest.mark.asyncio
async def test_confirmation_tool_callbacks_shell_pattern() -> None:
    callbacks = ConfirmationToolCallbacks(
        tool_confirmation_patterns=[],
        shell_confirmation_patterns=[r"danger"],
    )
    tool_name = "shell_execute"
    command = "danger cmd"
    args = {"command": command}
    prompt = f"Execute shell command `{command}` for tool `{tool_name}`?"
    ui = make_ui_mock(confirm_sequence=[(prompt, False), (prompt, True)])

    # Denied
    res = await callbacks.before_tool_execution(
        context_name="Agent",
        tool_call_id="1",
        tool_name=tool_name,
        arguments=args,
        ui=ui,
    )
    assert isinstance(res, TextResult)
    assert res.content == "Shell command execution denied."

    res2 = await callbacks.before_tool_execution(
        context_name="Agent",
        tool_call_id="2",
        tool_name=tool_name,
        arguments=args,
        ui=ui,
    )
    assert res2 is None


@pytest.mark.asyncio
async def test_confirmation_tool_callbacks_both_patterns_shell_tool_two_prompts() -> None:
    tool_name = "shell_execute"
    command = "do something risky"
    args = {"command": command}
    tool_prompt = f"Execute tool `{tool_name}` with arguments `{args}`?"
    shell_prompt = f"Execute shell command `{command}` for tool `{tool_name}`?"
    ui = make_ui_mock(confirm_sequence=[(tool_prompt, True), (shell_prompt, False)])

    callbacks = ConfirmationToolCallbacks(
        tool_confirmation_patterns=[r"shell_execute"],
        shell_confirmation_patterns=[r"risky"],
    )

    res = await callbacks.before_tool_execution(
        context_name="Agent",
        tool_call_id="1",
        tool_name=tool_name,
        arguments=args,
        ui=ui,
    )
    assert isinstance(res, TextResult)
    assert res.content == "Shell command execution denied."
