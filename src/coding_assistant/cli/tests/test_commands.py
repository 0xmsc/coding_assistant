from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from coding_assistant.cli.commands import (
    CompactCommand,
    ExitCommand,
    HelpCommand,
    ImageCommand,
    execute_slash_command,
    get_default_commands,
)


@pytest.mark.asyncio
async def test_exit_command() -> None:
    session = Mock()
    cmd = ExitCommand()
    assert cmd.name == "/exit"
    assert "Exit" in cmd.description
    assert await cmd.execute(session=session, args="") is True


@pytest.mark.asyncio
async def test_compact_command_enqueues_compaction_prompt() -> None:
    session = Mock()
    session.enqueue_prompt = AsyncMock(return_value=True)
    cmd = CompactCommand()
    assert cmd.name == "/compact"

    result = await cmd.execute(session=session, args="")
    assert result is False
    session.enqueue_prompt.assert_awaited_once_with(
        "Immediately compact our conversation so far by using the `compact_conversation` tool.",
    )


@pytest.mark.asyncio
async def test_image_command_requires_path_or_url() -> None:
    session = Mock()
    cmd = ImageCommand()
    assert cmd.name == "/image"
    assert cmd.usage == "<path-or-url>"

    with patch("coding_assistant.cli.commands.rich_print") as mock_print:
        result = await cmd.execute(session=session, args="   ")
        assert result is False
        mock_print.assert_called_once()


@pytest.mark.asyncio
async def test_image_command_loads_image_and_enqueues_prompt() -> None:
    session = Mock()
    session.enqueue_prompt = AsyncMock(return_value=True)
    cmd = ImageCommand()

    with patch("coding_assistant.cli.commands.get_image", new=AsyncMock(return_value="data:image/png;base64,abc")):
        result = await cmd.execute(session=session, args="path/to/img.png")
        assert result is False
        session.enqueue_prompt.assert_awaited_once_with(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}],
        )


@pytest.mark.asyncio
async def test_image_command_handles_image_load_failure() -> None:
    session = Mock()
    cmd = ImageCommand()

    with (
        patch("coding_assistant.cli.commands.get_image", new=AsyncMock(side_effect=RuntimeError("Download failed"))),
        patch("coding_assistant.cli.commands.rich_print") as mock_print,
    ):
        result = await cmd.execute(session=session, args="https://example.com/bad.png")
        assert result is False
        mock_print.assert_called_once()
        assert "Failed to load image: Download failed" in mock_print.call_args[0][0]


@pytest.mark.asyncio
async def test_help_command_prints_registered_commands() -> None:
    session = Mock()
    cmd = HelpCommand()
    assert cmd.name == "/help"

    with patch("coding_assistant.cli.commands.rich_print") as mock_print:
        result = await cmd.execute(session=session, args="")
        assert result is False
        mock_print.assert_called_once()
        help_output = mock_print.call_args[0][0]
        assert "/exit" in help_output
        assert "/help" in help_output
        assert "/compact" in help_output
        assert "/image" in help_output


def test_get_default_commands() -> None:
    commands = get_default_commands()
    names = [c.name for c in commands]
    assert names == ["/exit", "/help", "/compact", "/image"]


@pytest.mark.asyncio
async def test_execute_slash_command_returns_none_for_non_command() -> None:
    session = Mock()
    commands = get_default_commands()
    result = await execute_slash_command(commands=commands, content="hello agent", session=session)
    assert result is None


@pytest.mark.asyncio
async def test_execute_slash_command_executes_matched_command() -> None:
    session = Mock()
    commands = get_default_commands()
    result = await execute_slash_command(commands=commands, content="/exit", session=session)
    assert result is True


@pytest.mark.asyncio
async def test_execute_slash_command_forwards_arguments() -> None:
    session = Mock()
    mock_cmd = Mock(spec=ImageCommand)
    mock_cmd.name = "/image"
    mock_cmd.execute = AsyncMock(return_value=False)

    result = await execute_slash_command(commands=[mock_cmd], content="/image path/to/file.png", session=session)
    assert result is False
    mock_cmd.execute.assert_awaited_once_with(session=session, args="path/to/file.png")


@pytest.mark.asyncio
async def test_execute_slash_command_handles_unknown_command() -> None:
    session = Mock()
    commands = get_default_commands()
    with patch("coding_assistant.cli.commands.rich_print") as mock_print:
        result = await execute_slash_command(commands=commands, content="/foobar", session=session)
        assert result is False
        mock_print.assert_called_once()
        assert "Unknown command" in mock_print.call_args[0][0]
