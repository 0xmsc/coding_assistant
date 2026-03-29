import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from rich.markdown import Markdown

from coding_assistant.app.cli import (
    DefaultAgentBundle,
    build_default_agent_config,
    handle_cli_input,
)
from coding_assistant.app.main import main, parse_args
from coding_assistant.app.output import DeltaRenderer, ParagraphBuffer, format_tool_call_display, print_tool_calls
from coding_assistant.app.session_app import (
    CLI_CONTROLLER,
    PromptSubmissionResult,
    SessionApp,
    _prompt_while_controller_is_active,
    _submit_prompt_or_warn,
    run_session_app,
)
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall


class FakeLocalToolRuntime:
    def __init__(self) -> None:
        self.local_endpoint: str | None = None
        self.closed = False

    def set_local_worker_endpoint(self, endpoint: str) -> None:
        self.local_endpoint = endpoint

    async def close(self) -> None:
        self.closed = True


class FakePromptSession:
    def __init__(self) -> None:
        self.controller_changed = asyncio.Event()

    async def wait_for_controller_change(self, *, controller: str) -> str:
        assert controller == CLI_CONTROLLER
        await self.controller_changed.wait()
        return "remote"


class FakePromptSubmissionSession:
    def __init__(self, result: PromptSubmissionResult) -> None:
        self.result = result
        self.submitted_content: list[str | list[dict[str, object]]] = []

    async def submit_prompt(self, *, controller: str, content: str | list[dict[str, object]]) -> PromptSubmissionResult:
        assert controller == CLI_CONTROLLER
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


@patch("coding_assistant.app.main.run_session_app")
@patch("coding_assistant.app.main.enable_tracing")
def test_main_enables_tracing_when_flag_set(mock_enable_tracing: Any, mock_run_session_app: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--trace"]):
        main()
        mock_enable_tracing.assert_called_once()
        mock_run_session_app.assert_called_once()


@patch("coding_assistant.app.main.run_session_app")
@patch("coding_assistant.app.main.debugpy.wait_for_client")
@patch("coding_assistant.app.main.debugpy.listen")
def test_main_waits_for_debugger(mock_listen: Any, mock_wait: Any, mock_run_session_app: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--wait-for-debugger"]):
        main()
        mock_listen.assert_called_once_with(1234)
        mock_wait.assert_called_once()
        mock_run_session_app.assert_called_once()


def test_help_exits_with_zero() -> None:
    with patch("sys.argv", ["coding-assistant", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            parse_args()
        assert exc_info.value.code == 0


@pytest.mark.asyncio
async def test_run_session_app_builds_session_app_with_system_history() -> None:
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

    captured: dict[str, SessionApp] = {}

    async def fake_run(self: SessionApp) -> None:
        captured["app"] = self

    with (
        patch("coding_assistant.app.session_app.create_default_agent", fake_create_default_agent),
        patch("coding_assistant.app.session_app.SessionApp.run", new=fake_run),
    ):
        await run_session_app(args)

    session_app = captured["app"]
    assert session_app.history == [
        SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions.")),
    ]
    assert session_app.state.controller == CLI_CONTROLLER
    assert local_tool_runtime.local_endpoint is None
    assert local_tool_runtime.closed is False


@pytest.mark.asyncio
async def test_session_app_run_prints_system_message_before_driving_cli(tmp_path: Any) -> None:
    local_tool_runtime = FakeLocalToolRuntime()
    system_message = SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions."))

    async def prompt_user(words: list[str] | None = None) -> str:
        del words
        return "/exit"

    app = SessionApp(
        system_message=system_message,
        history=[system_message],
        model="gpt-4",
        tools=[],
        prompt_user=prompt_user,
        working_directory=tmp_path,
        set_local_worker_endpoint=local_tool_runtime.set_local_worker_endpoint,
        close_tools=local_tool_runtime.close,
    )

    @asynccontextmanager
    async def fake_start_worker_server(*, session: Any, cwd: Any) -> Any:
        assert session is app
        assert cwd == tmp_path
        yield SimpleNamespace(endpoint="ws://127.0.0.1:43210")

    captured: dict[str, SessionApp] = {}

    async def fake_drive_cli(self: SessionApp) -> None:
        captured["app"] = self

    with (
        patch("coding_assistant.remote.server.start_worker_server", fake_start_worker_server),
        patch("coding_assistant.app.session_app.print_system_message") as mock_print_system,
        patch.object(SessionApp, "_drive_cli", new=fake_drive_cli),
    ):
        await app.run()

    mock_print_system.assert_called_once_with(system_message)
    assert captured["app"] is app
    assert local_tool_runtime.local_endpoint == "ws://127.0.0.1:43210"
    assert local_tool_runtime.closed is True


@pytest.mark.asyncio
async def test_prompt_without_remote_controller_returns_prompt_when_user_finishes_first() -> None:
    session = FakePromptSession()

    async def prompt_user(words: list[str] | None) -> str:
        assert words == ["/exit"]
        return "hello"

    result = await _prompt_while_controller_is_active(
        session=session,  # type: ignore[arg-type]
        controller=CLI_CONTROLLER,
        prompt_user=prompt_user,
        words=["/exit"],
    )

    assert result == "hello"


@pytest.mark.asyncio
async def test_prompt_without_remote_controller_returns_none_when_remote_connects_first() -> None:
    session = FakePromptSession()
    prompt_started = asyncio.Event()

    async def prompt_user(words: list[str] | None) -> str:
        del words
        prompt_started.set()
        await asyncio.Future()
        return ""

    task = asyncio.create_task(
        _prompt_while_controller_is_active(
            session=session,  # type: ignore[arg-type]
            controller=CLI_CONTROLLER,
            prompt_user=prompt_user,
            words=["/exit"],
        )
    )
    await prompt_started.wait()
    session.controller_changed.set()

    assert await asyncio.wait_for(task, timeout=1) is None


@pytest.mark.asyncio
async def test_submit_local_prompt_or_warn_reports_remote_takeover() -> None:
    session = FakePromptSubmissionSession(PromptSubmissionResult(accepted=False, reason="inactive_controller"))

    with patch("coding_assistant.app.session_app.print") as mock_print:
        result = await _submit_prompt_or_warn(
            session=session,  # type: ignore[arg-type]
            controller=CLI_CONTROLLER,
            content="hello",
        )

    assert result is False
    assert session.submitted_content == ["hello"]
    mock_print.assert_called_once_with("Remote control took ownership before the local prompt was submitted.")


@pytest.mark.asyncio
async def test_handle_cli_input_prints_help_without_submitting() -> None:
    renderer = AsyncMock()
    submitted: list[str | list[dict[str, object]]] = []

    async def fake_submit(content: str | list[dict[str, object]]) -> bool:
        submitted.append(content)
        return True

    with patch("coding_assistant.app.cli.print") as mock_print:
        result = await handle_cli_input(
            answer="/help",
            renderer=renderer,
            submit_prompt_or_warn=fake_submit,
        )

    assert result is False
    assert submitted == []
    mock_print.assert_called_once_with("Available commands:\n  /exit\n  /help\n  /compact\n  /image <path-or-url>")


@pytest.mark.asyncio
async def test_handle_cli_input_compact_submits_compaction_prompt() -> None:
    renderer = AsyncMock()
    submitted: list[str | list[dict[str, object]]] = []

    async def fake_submit(content: str | list[dict[str, object]]) -> bool:
        submitted.append(content)
        return True

    result = await handle_cli_input(
        answer="/compact",
        renderer=renderer,
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
