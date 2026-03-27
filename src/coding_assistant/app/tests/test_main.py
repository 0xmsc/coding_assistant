from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.markdown import Markdown

from coding_assistant.app.cli import (
    DefaultAgentBundle,
    DeltaRenderer,
    ParagraphBuffer,
    _format_tool_call_display,
    _drive_agent,
    build_default_agent_config,
    run_cli,
)
from coding_assistant.core.agent import AwaitingTools, AwaitingUser
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall, UserMessage
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


def test_paragraph_buffer_respects_code_fences() -> None:
    buffer = ParagraphBuffer()

    assert buffer.push("Before\n\n```python\nprint('hi')") == ["Before"]
    assert buffer.push("\n\nprint('bye')\n```\n\nAfter") == ["```python\nprint('hi')\n\nprint('bye')\n```"]
    assert buffer.flush() == "After"


def test_delta_renderer_prints_markdown_paragraphs() -> None:
    renderer = DeltaRenderer()

    with patch("coding_assistant.app.cli.rich_print") as mock_print:
        renderer.on_delta("First paragraph")
        renderer.on_delta("\n\nSecond paragraph")
        renderer.finish()

    markdown_blocks = [
        call.args[0] for call in mock_print.call_args_list if call.args and isinstance(call.args[0], Markdown)
    ]
    assert [block.markup for block in markdown_blocks] == ["First paragraph", "Second paragraph"]


def test_format_tool_call_markdown_formats_multiline_arguments() -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(
            name="shell_execute",
            arguments='{"command": "echo hello\\npwd", "background": false}',
        ),
    )

    header, body_sections = _format_tool_call_display(tool_call)

    assert header == "shell_execute(command, background=false)"
    assert body_sections == [("command", "echo hello\npwd", "bash")]


def test_format_tool_call_markdown_hides_edit_payload_values() -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(
            name="filesystem_edit_file",
            arguments='{"path": "script.sh", "old_text": "old", "new_text": "new"}',
        ),
    )

    header, body_sections = _format_tool_call_display(tool_call)

    assert header == 'filesystem_edit_file(path="script.sh", old_text, new_text)'
    assert body_sections == []


@pytest.mark.asyncio
async def test_drive_agent_prints_formatted_tool_call_before_execution() -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments='{"count": 2}'),
    )
    tool_boundary = AwaitingTools(
        history=[
            SystemMessage(content="System"),
            UserMessage(content="Do the task"),
            AssistantMessage(tool_calls=[tool_call]),
        ],
        message=AssistantMessage(tool_calls=[tool_call]),
    )

    with (
        patch(
            "coding_assistant.app.cli.run_agent_until_boundary",
            new=AsyncMock(
                side_effect=[
                    tool_boundary,
                    AwaitingUser(history=[SystemMessage(content="System"), AssistantMessage(content="Done")]),
                ]
            ),
        ),
        patch("coding_assistant.app.cli.execute_tool_calls", new=AsyncMock(return_value=tool_boundary.history)),
        patch("coding_assistant.app.cli.rich_print") as mock_print,
    ):
        await _drive_agent(
            history=[SystemMessage(content="System"), UserMessage(content="Do the task")],
            model="gpt-4",
            tools=[],
            ui=Mock(),
            interactive=False,
        )

    assert any(call.args == ("[bold yellow]▶[/bold yellow] mock_tool(count=2)",) for call in mock_print.call_args_list)
