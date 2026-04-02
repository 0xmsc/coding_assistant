from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from coding_assistant.app.cli import _handle_prompt_submission, run_cli
from coding_assistant.app.terminal_ui import PromptSubmitType
from coding_assistant.app.default_agent import DefaultAgentBundle, build_default_agent_config
from coding_assistant.app.main import main, parse_args
from coding_assistant.app.output import (
    DeltaRenderer,
    ParagraphBuffer,
    format_session_status,
    format_tool_call_display,
    print_tool_calls,
)
from coding_assistant.core.agent_session import AgentSession, SessionState
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall
from coding_assistant.remote.server import WorkerServer


def test_parse_args_valid() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.model == "gpt-4"


def test_parse_args_defaults() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.skills_directories == []


def test_parse_args_with_multiple_flags() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--trace"]):
        args = parse_args()
        assert args.trace is True


def test_build_default_agent_config_from_args(tmp_path: Any) -> None:
    args = type("MockArgs", (), {})()
    args.mcp_servers = []
    args.skills_directories = []
    args.instructions = []

    with patch("coding_assistant.app.default_agent.os.getcwd", return_value=str(tmp_path)):
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


@patch("coding_assistant.app.main.run_cli")
def test_main_runs_cli(mock_run_cli: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model"]):
        main()
        mock_run_cli.assert_called_once()


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
    )

    @asynccontextmanager
    async def fake_create_default_agent(*, config: Any) -> Any:
        del config
        yield DefaultAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
            mcp_servers=[],
        )

    @asynccontextmanager
    async def fake_start_worker_server(*, session: Any) -> Any:
        yield WorkerServer(endpoint="ws://127.0.0.1:1234")

    @asynccontextmanager
    async def fake_register_remote_instance(*, endpoint: str) -> Any:
        assert endpoint == "ws://127.0.0.1:1234"
        yield

    with (
        patch("coding_assistant.app.cli.create_default_agent", fake_create_default_agent),
        patch("coding_assistant.app.cli.start_worker_server", fake_start_worker_server),
        patch("coding_assistant.app.cli.register_remote_instance", fake_register_remote_instance),
        patch("coding_assistant.app.cli.print") as mock_print,
        patch("coding_assistant.app.cli.run_terminal_ui", new=AsyncMock()) as mock_run_terminal_ui,
    ):
        await run_cli(args)

    assert mock_run_terminal_ui.await_args is not None
    session = mock_run_terminal_ui.await_args.kwargs["session"]
    assert isinstance(session, AgentSession)
    assert session.history == [
        SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions.")),
    ]
    assert mock_run_terminal_ui.await_args.kwargs["system_message"] == SystemMessage(
        content=build_system_prompt(instructions="Follow the repo instructions.")
    )
    assert mock_run_terminal_ui.await_args.kwargs["history_path"].name == "history"
    assert mock_run_terminal_ui.await_args.kwargs["words"] == [
        "/exit",
        "/help",
        "/compact",
        "/image",
        "/priority",
        "/interrupt",
    ]
    mock_print.assert_called_once_with("Remote endpoint: ws://127.0.0.1:1234")


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


def test_format_session_status_summarizes_pending_prompts() -> None:
    state = SessionState(
        running=True,
        queued_prompt_count=3,
        pending_prompts=("first queued prompt", "second queued prompt", "third queued prompt"),
    )

    # When pending prompts exist, they're shown in the queued prompts widget above the input.
    # The footer only shows the status to avoid redundancy.
    assert format_session_status(state) == "running"


def test_print_prompt_accepted_uses_simple_grey_background() -> None:
    with patch("coding_assistant.app.output.rich_print") as mock_print:
        from coding_assistant.app.output import print_active_prompt

        print_active_prompt("Do the task")

    assert len(mock_print.call_args_list) == 1
    group = mock_print.call_args_list[0].args[0]
    assert isinstance(group, Group)
    assert len(group.renderables) == 1
    line = group.renderables[0]
    assert isinstance(line, Text)
    assert line.plain == "▌ Do the task"
    assert [(span.start, span.end, span.style) for span in line.spans] == [
        (0, 2, "grey50"),
        (2, 13, "on grey11"),
    ]


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
async def test_handle_prompt_submission_enqueues_steering_prompt() -> None:
    session = Mock()
    session.enqueue_steering_prompt = AsyncMock(return_value=True)

    should_exit = await _handle_prompt_submission(
        session=session, answer="fix this next", submit_type=PromptSubmitType.STEERING
    )

    assert should_exit is False
    session.enqueue_steering_prompt.assert_awaited_once_with("fix this next")


@pytest.mark.asyncio
async def test_handle_prompt_submission_queues_priority_prompt() -> None:
    session = Mock()
    session.enqueue_prompt = AsyncMock(return_value=True)

    should_exit = await _handle_prompt_submission(
        session=session, answer="/priority fix this next", submit_type=PromptSubmitType.QUEUED
    )

    assert should_exit is False
    session.enqueue_prompt.assert_awaited_once_with("fix this next", priority=True)


@pytest.mark.asyncio
async def test_handle_prompt_submission_interrupts_current_run() -> None:
    session = Mock()
    session.interrupt_and_enqueue = AsyncMock(return_value=True)

    should_exit = await _handle_prompt_submission(
        session=session, answer="/interrupt stop and do this instead", submit_type=PromptSubmitType.QUEUED
    )

    assert should_exit is False
    session.interrupt_and_enqueue.assert_awaited_once_with("stop and do this instead")


@pytest.mark.asyncio
async def test_handle_prompt_submission_compact_enqueues_compaction_prompt() -> None:
    session = Mock()
    session.enqueue_prompt = AsyncMock(return_value=True)

    should_exit = await _handle_prompt_submission(
        session=session, answer="/compact", submit_type=PromptSubmitType.QUEUED
    )

    assert should_exit is False
    session.enqueue_prompt.assert_awaited_once_with(
        "Immediately compact our conversation so far by using the `compact_conversation` tool."
    )
