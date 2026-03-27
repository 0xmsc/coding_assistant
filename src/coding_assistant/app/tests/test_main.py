from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from coding_assistant.app.cli import DefaultAgentBundle, build_default_agent_config, run_cli
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import SystemMessage, UserMessage
from coding_assistant.app.main import main, parse_args


def test_parse_args_valid() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--task", "test"]):
        args = parse_args()
        assert args.model == "gpt-4"
        assert args.task == "test"
        assert args.sandbox is True


def test_parse_args_defaults() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.ask_user is True


def test_parse_args_with_multiple_flags() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--trace", "--no-sandbox", "--no-ask-user"]):
        args = parse_args()
        assert args.trace is True
        assert args.sandbox is False
        assert args.ask_user is False


def test_build_default_agent_config_from_args(tmp_path: Any) -> None:
    args = type("MockArgs", (), {})()
    args.mcp_servers = []
    args.skills_directories = []
    args.mcp_env = []
    args.instructions = []

    with patch("coding_assistant.app.cli.os.getcwd", return_value=str(tmp_path)):
        config = build_default_agent_config(args)

    assert config.working_directory == tmp_path
    assert config.skills_directories == ()
    assert config.mcp_env == ()
    assert config.user_instructions == ()


@patch("coding_assistant.app.main.run_cli")
@patch("coding_assistant.app.main.enable_tracing")
def test_main_enables_tracing_when_flag_set(mock_enable_tracing: Any, mock_run_cli: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--trace"]):
        main()
        mock_enable_tracing.assert_called_once()
        mock_run_cli.assert_called_once()


@patch("coding_assistant.app.main.run_cli")
@patch("coding_assistant.app.main.debugpy.wait_for_client")
@patch("coding_assistant.app.main.debugpy.listen")
def test_main_waits_for_debugger(mock_listen: Any, mock_wait: Any, mock_run_cli: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--wait-for-debugger"]):
        main()
        mock_listen.assert_called_once_with(1234)
        mock_wait.assert_called_once()
        mock_run_cli.assert_called_once()


def test_parse_args_mcp_env() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--mcp-env", "VAR1", "VAR2"]):
        args = parse_args()
        assert args.mcp_env == ["VAR1", "VAR2"]


def test_help_exits_with_zero() -> None:
    with patch("sys.argv", ["coding-assistant", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 0


@pytest.mark.asyncio
async def test_run_cli_prints_system_message_before_running_agent() -> None:
    args = Namespace(
        ask_user=False,
        instructions=[],
        mcp_env=[],
        mcp_servers=[],
        model="gpt-4",
        print_mcp_tools=False,
        readable_sandbox_directories=[],
        sandbox=False,
        skills_directories=[],
        task="test task",
        writable_sandbox_directories=[],
    )

    @asynccontextmanager
    async def fake_create_default_agent(*, config: Any) -> Any:
        del config
        yield DefaultAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
            mcp_servers=[],
        )

    with (
        patch("coding_assistant.app.cli.create_default_agent", fake_create_default_agent),
        patch("coding_assistant.app.cli.rich_print") as mock_print,
        patch("coding_assistant.app.cli._drive_agent", new=AsyncMock()) as mock_drive_agent,
    ):
        await run_cli(args)

    mock_print.assert_called_once()
    assert mock_drive_agent.await_args is not None
    history = mock_drive_agent.await_args.kwargs["history"]
    assert history == [
        SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions.")),
        UserMessage(content="test task"),
    ]
