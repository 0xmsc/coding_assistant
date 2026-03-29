import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import pytest
from rich.markdown import Markdown

from coding_assistant.app.cli import (
    CliController,
    CliOutput,
    _prompt_while_controller_is_active,
    _submit_prompt_or_warn,
    DefaultAgentBundle,
    build_default_agent_config,
    handle_cli_input,
)
from coding_assistant.app.main import main, parse_args, run_session_runtime
from coding_assistant.app.output import DeltaRenderer, ParagraphBuffer, format_tool_call_display, print_tool_calls
from coding_assistant.app.session_control import PromptSubmissionResult
from coding_assistant.app.session_runtime import SessionRuntime
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall
from coding_assistant.remote.server import RemoteController


class FakeLocalToolRuntime:
    def __init__(self) -> None:
        self.local_endpoint: str | None = None
        self.closed = False

    def set_local_worker_endpoint(self, endpoint: str) -> None:
        self.local_endpoint = endpoint

    async def close(self) -> None:
        self.closed = True


class FakePromptSession:
    def __init__(self, expected_controller: object) -> None:
        self.expected_controller = expected_controller
        self.controller_changed = asyncio.Event()

    async def wait_for_controller_change(self, *, controller: object) -> object:
        assert controller is self.expected_controller
        await self.controller_changed.wait()
        return object()


class FakePromptSubmissionSession:
    def __init__(self, result: PromptSubmissionResult, expected_controller: object) -> None:
        self.result = result
        self.expected_controller = expected_controller
        self.submitted_content: list[str | list[dict[str, object]]] = []

    async def submit_prompt(
        self, *, controller: object, content: str | list[dict[str, object]]
    ) -> PromptSubmissionResult:
        assert controller is self.expected_controller
        self.submitted_content.append(content)
        return self.result


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

    with patch("coding_assistant.app.cli.os.getcwd", return_value=str(tmp_path)):
        config = build_default_agent_config(args)

    assert config.working_directory == tmp_path
    assert config.skills_directories == ()
    assert config.user_instructions == ()


@patch("coding_assistant.app.main.run_session_runtime")
@patch("coding_assistant.app.main.enable_tracing")
def test_main_enables_tracing_when_flag_set(mock_enable_tracing: Any, mock_run_session_runtime: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--trace"]):
        main()
        mock_enable_tracing.assert_called_once()
        mock_run_session_runtime.assert_called_once()


@patch("coding_assistant.app.main.run_session_runtime")
@patch("coding_assistant.app.main.debugpy.wait_for_client")
@patch("coding_assistant.app.main.debugpy.listen")
def test_main_waits_for_debugger(mock_listen: Any, mock_wait: Any, mock_run_session_runtime: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--wait-for-debugger"]):
        main()
        mock_listen.assert_called_once_with(1234)
        mock_wait.assert_called_once()
        mock_run_session_runtime.assert_called_once()


def test_help_exits_with_zero() -> None:
    with patch("sys.argv", ["coding-assistant", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 0


@pytest.mark.asyncio
async def test_run_session_runtime_builds_runtime_with_default_adapters() -> None:
    args = Namespace(
        instructions=[],
        mcp_servers=[],
        model="gpt-4",
        print_mcp_tools=False,
        skills_directories=[],
        trace=False,
        wait_for_debugger=False,
    )
    local_tool_runtime = FakeLocalToolRuntime()

    @asynccontextmanager
    async def fake_create_default_agent(*, config: Any) -> Any:
        del config
        yield DefaultAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
            mcp_servers=[],
            set_local_worker_endpoint=local_tool_runtime.set_local_worker_endpoint,
            close_tools=local_tool_runtime.close,
        )

    captured_runtime: SessionRuntime | None = None
    captured_controllers: list[object] = []
    captured_outputs: list[object] = []

    async def fake_run(self: SessionRuntime, *, controllers: Any, outputs: Any) -> None:
        nonlocal captured_runtime, captured_controllers, captured_outputs
        captured_runtime = self
        captured_controllers = list(controllers)
        captured_outputs = list(outputs)

    with (
        patch("coding_assistant.app.main.create_default_agent", fake_create_default_agent),
        patch("coding_assistant.app.main.SessionRuntime.run", new=fake_run),
    ):
        await run_session_runtime(args)

    session_runtime = captured_runtime
    assert isinstance(session_runtime, SessionRuntime)
    assert session_runtime.history == [
        SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions.")),
    ]
    assert len(captured_controllers) == 2
    assert isinstance(captured_controllers[0], CliController)
    assert isinstance(captured_controllers[1], RemoteController)
    assert session_runtime.is_active_controller(captured_controllers[0]) is True
    assert len(captured_outputs) == 1
    assert isinstance(captured_outputs[0], CliOutput)
    assert local_tool_runtime.local_endpoint is None
    assert local_tool_runtime.closed is True


@pytest.mark.asyncio
async def test_prompt_without_remote_controller_returns_prompt_when_user_finishes_first() -> None:
    controller = object()
    session = FakePromptSession(expected_controller=controller)

    async def prompt_user(words: list[str] | None) -> str:
        assert words == ["/exit"]
        return "hello"

    result = await _prompt_while_controller_is_active(
        session=session,  # type: ignore[arg-type]
        controller=controller,  # type: ignore[arg-type]
        prompt_user=prompt_user,
        words=["/exit"],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_prompt_without_remote_controller_returns_none_when_remote_connects_first() -> None:
    controller = object()
    session = FakePromptSession(expected_controller=controller)
    prompt_started = asyncio.Event()

    async def prompt_user(words: list[str] | None) -> str:
        del words
        prompt_started.set()
        await asyncio.Future()
        return ""

    task = asyncio.create_task(
        _prompt_while_controller_is_active(
            session=session,  # type: ignore[arg-type]
            controller=controller,  # type: ignore[arg-type]
            prompt_user=prompt_user,
            words=["/exit"],
        )
    )
    await prompt_started.wait()
    session.controller_changed.set()

    assert await asyncio.wait_for(task, timeout=1) is None


@pytest.mark.asyncio
async def test_submit_local_prompt_or_warn_reports_remote_takeover() -> None:
    controller = object()
    session = FakePromptSubmissionSession(
        PromptSubmissionResult(accepted=False, reason="inactive_controller"),
        expected_controller=controller,
    )

    with patch("coding_assistant.app.cli.print") as mock_print:
        result = await _submit_prompt_or_warn(
            session=session,  # type: ignore[arg-type]
            controller=controller,  # type: ignore[arg-type]
            content="hello",
        )

    assert result is False
    assert session.submitted_content == ["hello"]
    mock_print.assert_called_once_with("Remote control took ownership before the local prompt was submitted.")


@pytest.mark.asyncio
async def test_handle_cli_input_prints_help_without_submitting() -> None:
    submitted: list[str | list[dict[str, object]]] = []

    async def fake_submit(content: str | list[dict[str, object]]) -> bool:
        submitted.append(content)
        return True

    with patch("coding_assistant.app.cli.print") as mock_print:
        result = await handle_cli_input(
            answer="/help",
            submit_prompt_or_warn=fake_submit,
        )

    assert result is False
    assert submitted == []
    mock_print.assert_called_once_with("Available commands:\n  /exit\n  /help\n  /compact\n  /image <path-or-url>")


@pytest.mark.asyncio
async def test_handle_cli_input_compact_submits_compaction_prompt() -> None:
    submitted: list[str | list[dict[str, object]]] = []

    async def fake_submit(content: str | list[dict[str, object]]) -> bool:
        submitted.append(content)
        return True

    result = await handle_cli_input(
        answer="/compact",
        submit_prompt_or_warn=fake_submit,
    )

    assert result is False
    assert submitted == ["Immediately compact our conversation so far by using the `compact_conversation` tool."]


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
