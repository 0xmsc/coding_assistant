from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.markdown import Markdown

from coding_assistant.app.cli import DefaultAgentBundle, _drive_agent, build_default_agent_config, run_cli
from coding_assistant.app.main import main, parse_args
from coding_assistant.app.output import DeltaRenderer, ParagraphBuffer, format_tool_call_display, print_tool_calls
from coding_assistant.core.boundaries import AwaitingToolCalls, AwaitingUser
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall, UserMessage


def test_parse_args_valid() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.model == "gpt-4"


def test_parse_args_defaults() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.skills_directories == []
        assert args.worker is False


def test_parse_args_with_multiple_flags() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--trace"]):
        args = parse_args()
        assert args.trace is True


def test_parse_args_worker_flag() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--worker"]):
        args = parse_args()
        assert args.worker is True


def test_build_default_agent_config_from_args(tmp_path: Any) -> None:
    args = type("MockArgs", (), {})()
    args.mcp_servers = []
    args.skills_directories = []
    args.instructions = []

    with patch("coding_assistant.app.cli.os.getcwd", return_value=str(tmp_path)):
        config = build_default_agent_config(args)

    assert config.working_directory == tmp_path
    assert config.skills_directories == ()
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


@patch("coding_assistant.app.main.run_worker")
@patch("coding_assistant.app.main.run_cli")
def test_main_dispatches_worker_mode(mock_run_cli: Any, mock_run_worker: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--worker"]):
        main()
        mock_run_worker.assert_called_once()
        mock_run_cli.assert_not_called()


def test_help_exits_with_zero() -> None:
    with patch("sys.argv", ["coding-assistant", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 0


@pytest.mark.asyncio
async def test_run_cli_prints_system_message_before_running_agent() -> None:
    args = Namespace(
        instructions=[],
        mcp_servers=[],
        model="gpt-4",
        print_mcp_tools=False,
        skills_directories=[],
        trace=False,
        wait_for_debugger=False,
        worker=False,
    )

    @asynccontextmanager
    async def fake_create_default_agent(*, config: Any, include_worker_tools: bool = True) -> Any:
        del config
        assert include_worker_tools is True
        yield DefaultAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
            mcp_servers=[],
        )

    with (
        patch("coding_assistant.app.cli.create_default_agent", fake_create_default_agent),
        patch("coding_assistant.app.cli.print_system_message") as mock_print_system,
        patch("coding_assistant.app.cli._drive_agent", new=AsyncMock()) as mock_drive_agent,
    ):
        await run_cli(args)

    mock_print_system.assert_called_once()
    assert mock_drive_agent.await_args is not None
    history = mock_drive_agent.await_args.kwargs["history"]
    assert history == [
        SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions.")),
    ]


def test_paragraph_buffer_respects_code_fences() -> None:
    buffer = ParagraphBuffer()

    assert buffer.push("Before\n\n```python\nprint('hi')") == ["Before"]
    assert buffer.push("\n\nprint('bye')\n```\n\nAfter") == ["```python\nprint('hi')\n\nprint('bye')\n```"]
    assert buffer.flush() == "After"


def test_delta_renderer_prints_markdown_paragraphs() -> None:
    renderer = DeltaRenderer()

    with patch("coding_assistant.app.output.rich_print") as mock_print:
        renderer.on_delta("First paragraph")
        renderer.on_delta("\n\nSecond paragraph")
        renderer.finish()

    markdown_blocks = [
        call.args[0] for call in mock_print.call_args_list if call.args and isinstance(call.args[0], Markdown)
    ]
    assert [block.markup for block in markdown_blocks] == ["First paragraph", "Second paragraph"]


def test_delta_renderer_avoids_double_spacing_before_tool_calls() -> None:
    renderer = DeltaRenderer()
    tool_call_message = AssistantMessage(
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(
                    name="shell_execute",
                    arguments='{"command": "cat README.md"}',
                ),
            )
        ]
    )

    with patch("coding_assistant.app.output.rich_print") as mock_print:
        renderer.on_delta("Can you read README.md?")
        renderer.finish(trailing_blank_line=False)
        print_tool_calls(tool_call_message)

    assert len(mock_print.call_args_list) == 4
    assert mock_print.call_args_list[0].args == ()
    assert isinstance(mock_print.call_args_list[1].args[0], Markdown)
    assert mock_print.call_args_list[1].args[0].markup == "Can you read README.md?"
    assert mock_print.call_args_list[2].args == ()
    assert mock_print.call_args_list[3].args == ('[bold yellow]▶[/bold yellow] shell_execute(command="cat README.md")',)


def test_delta_renderer_finish_is_idempotent() -> None:
    renderer = DeltaRenderer()

    with patch("coding_assistant.app.output.rich_print") as mock_print:
        renderer.on_delta("Hello")
        renderer.finish()
        renderer.finish()

    assert len(mock_print.call_args_list) == 3
    assert mock_print.call_args_list[0].args == ()
    assert isinstance(mock_print.call_args_list[1].args[0], Markdown)
    assert mock_print.call_args_list[1].args[0].markup == "Hello"
    assert mock_print.call_args_list[2].args == ()


def test_format_tool_call_markdown_formats_multiline_arguments() -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(
            name="shell_execute",
            arguments='{"command": "echo hello\\npwd", "background": false}',
        ),
    )

    header, body_sections = format_tool_call_display(tool_call)

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

    header, body_sections = format_tool_call_display(tool_call)

    assert header == 'filesystem_edit_file(path="script.sh", old_text, new_text)'
    assert body_sections == []


@pytest.mark.asyncio
async def test_drive_agent_prints_formatted_tool_call_before_execution() -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="mock_tool", arguments='{"count": 2}'),
    )
    tool_boundary = AwaitingToolCalls(
        history=[
            SystemMessage(content="System"),
            UserMessage(content="Do the task"),
            AssistantMessage(tool_calls=[tool_call]),
        ],
    )

    boundaries: list[AwaitingUser | AwaitingToolCalls] = [
        tool_boundary,
        AwaitingUser(history=[SystemMessage(content="System"), AssistantMessage(content="Done")]),
    ]

    async def fake_run_agent_event_stream(**kwargs: Any) -> AsyncIterator[AwaitingUser | AwaitingToolCalls]:
        del kwargs
        yield boundaries.pop(0)

    with (
        patch(
            "coding_assistant.app.cli.run_agent_event_stream",
            new=fake_run_agent_event_stream,
        ),
        patch("coding_assistant.app.cli.execute_tool_calls", new=AsyncMock(return_value=tool_boundary.history)),
        patch("coding_assistant.app.output.rich_print") as mock_print,
    ):
        ui = Mock()
        ui.prompt = AsyncMock(return_value="/exit")
        await _drive_agent(
            history=[SystemMessage(content="System"), UserMessage(content="Do the task")],
            model="gpt-4",
            tools=[],
            prompt_user=ui.prompt,
        )

    assert any(call.args == ("[bold yellow]▶[/bold yellow] mock_tool(count=2)",) for call in mock_print.call_args_list)
