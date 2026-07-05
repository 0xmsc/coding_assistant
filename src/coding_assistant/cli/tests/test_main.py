import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.markdown import Markdown

from coding_assistant.cli.ui import _handle_submission, PromptSubmitType, run_cli
from coding_assistant.cli.agent import (
    CliAgentBundle,
    build_cli_agent_config,
    create_cli_agent,
)
from coding_assistant.cli.main import main, parse_args
from coding_assistant.cli.output import (
    DeltaRenderer,
    ParagraphBuffer,
    format_session_status,
    print_tool_calls,
)
from coding_assistant.core.agent_session import AgentSession, RunFinishedEvent, SessionState
from coding_assistant.core.runtime import build_initial_system_message
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall, UserMessage
from coding_assistant.remote.server import RemoteServer
from coding_assistant.testing.fake_openai import run_fake_openai_server


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


def test_build_cli_agent_config_from_args(tmp_path: Any) -> None:
    args = type("MockArgs", (), {})()
    args.mcp_servers = []
    args.skills_directories = []
    args.instructions = []

    with patch("coding_assistant.cli.agent.os.getcwd", return_value=str(tmp_path)):
        config = build_cli_agent_config(args)

    assert config.working_directory == tmp_path
    assert config.skills_directories == ()
    assert config.user_instructions == ()


@patch("coding_assistant.cli.main.run_cli")
@patch("coding_assistant.cli.main.enable_tracing")
def test_main_enables_tracing_when_flag_set(mock_enable_tracing: Any, mock_run_cli: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--trace"]):
        main()
        mock_enable_tracing.assert_called_once()
        mock_run_cli.assert_called_once()


@patch("coding_assistant.cli.main.run_cli")
@patch("coding_assistant.cli.main.debugpy.wait_for_client")
@patch("coding_assistant.cli.main.debugpy.listen")
def test_main_waits_for_debugger(mock_listen: Any, mock_wait: Any, mock_run_cli: Any) -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "test-model", "--wait-for-debugger"]):
        main()
        mock_listen.assert_called_once_with(1234)
        mock_wait.assert_called_once()
        mock_run_cli.assert_called_once()


@patch("coding_assistant.cli.main.run_cli")
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
        skills_directories=[],
        trace=False,
        wait_for_debugger=False,
    )

    @asynccontextmanager
    async def fake_create_cli_agent(*, config: Any) -> Any:
        del config
        yield CliAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
        )

    @asynccontextmanager
    async def fake_start_worker_server(*, session: Any) -> Any:
        yield RemoteServer(endpoint="ws://127.0.0.1:1234")

    @asynccontextmanager
    async def fake_register_remote_instance(*, endpoint: str) -> Any:
        assert endpoint == "ws://127.0.0.1:1234"
        yield

    with (
        patch("coding_assistant.cli.ui.create_cli_agent", fake_create_cli_agent),
        patch("coding_assistant.cli.ui.start_worker_server", fake_start_worker_server),
        patch("coding_assistant.cli.ui.register_remote_instance", fake_register_remote_instance),
        patch("coding_assistant.cli.ui.rich_print") as mock_rich_print,
        patch("coding_assistant.cli.ui._run_ui", new=AsyncMock()) as mock_run_ui,
    ):
        await run_cli(args)

    assert mock_run_ui.await_args is not None
    session = mock_run_ui.await_args.kwargs["session"]
    assert isinstance(session, AgentSession)
    assert session.history == [
        SystemMessage(content="Follow the repo instructions."),
    ]
    assert mock_run_ui.await_args.kwargs["system_message"] == SystemMessage(
        content="Follow the repo instructions.",
    )
    assert mock_run_ui.await_args.kwargs["history_path"].name == "history"
    mock_rich_print.assert_called_once()


@pytest.mark.asyncio
async def test_cli_agent_smoke_runs_against_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    args = Namespace(
        instructions=[],
        mcp_servers=[],
        model="fake-model",
        skills_directories=[],
        trace=False,
        wait_for_debugger=False,
    )
    monkeypatch.setattr("coding_assistant.cli.agent.os.getcwd", lambda: str(tmp_path))
    monkeypatch.setenv("CODING_ASSISTANT_FAKE_OPENAI_RESPONSE", "cli agent smoke response")

    with run_fake_openai_server() as fake_openai:
        monkeypatch.setenv("OPENAI_BASE_URL", fake_openai.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        config = build_cli_agent_config(args)
        async with create_cli_agent(config=config) as bundle:
            system_message = build_initial_system_message(instructions=bundle.instructions)
            session = AgentSession(
                history=[system_message],
                model=args.model,
                tools=bundle.tools,
            )
            try:
                async with session.subscribe() as queue:
                    assert await session.enqueue_prompt("Run the fake smoke test.") is True
                    while True:
                        event = await asyncio.wait_for(queue.get(), timeout=2)
                        if isinstance(event, RunFinishedEvent):
                            break
            finally:
                await session.close()

    assert session.history[-2:] == [
        UserMessage(content="Run the fake smoke test."),
        AssistantMessage(
            content="cli agent smoke response",
            provider_specific_fields={"reasoning_details": []},
        ),
    ]


def test_paragraph_buffer_respects_code_fences() -> None:
    buffer = ParagraphBuffer()

    assert buffer.push("Before\n\n```python\nprint('hi')") == ["Before"]
    assert buffer.push("\n\nprint('bye')\n```\n\nAfter") == ["```python\nprint('hi')\n\nprint('bye')\n```"]
    assert buffer.flush() == "After"


def test_delta_renderer_prints_markdown_paragraphs() -> None:
    renderer = DeltaRenderer()

    with patch("coding_assistant.cli.output.rich_print") as mock_print:
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
            ),
        ],
    )

    with patch("coding_assistant.cli.output.rich_print") as mock_print:
        renderer.on_delta("Can you read README.md?")
        renderer.finish()
        print_tool_calls(tool_call_message)

    # Each element adds its own spacing:
    # 1. blank line before content (from _print_markdown)
    # 2. markdown content
    # 3. blank line before tool call (from print_tool_calls)
    # 4. tool call line
    assert len(mock_print.call_args_list) == 4
    assert mock_print.call_args_list[0].args == ()
    assert isinstance(mock_print.call_args_list[1].args[0], Markdown)
    assert mock_print.call_args_list[1].args[0].markup == "Can you read README.md?"
    assert mock_print.call_args_list[2].args == ()  # blank line before tool call
    assert "▶" in str(mock_print.call_args_list[3])
    assert "shell_execute" in str(mock_print.call_args_list[3])


def test_delta_renderer_finish_is_idempotent() -> None:
    renderer = DeltaRenderer()

    with patch("coding_assistant.cli.output.rich_print") as mock_print:
        renderer.on_delta("Hello")
        renderer.finish()
        renderer.finish()

    # Should call rich_print 2 times: blank line + content (no trailing blank)
    assert len(mock_print.call_args_list) == 2
    assert mock_print.call_args_list[0].args == ()
    assert isinstance(mock_print.call_args_list[1].args[0], Markdown)
    assert mock_print.call_args_list[1].args[0].markup == "Hello"


def test_format_session_status_summarizes_pending_prompts() -> None:
    state = SessionState(
        running=True,
        pending_prompts=("first queued prompt", "second queued prompt", "third queued prompt"),
    )

    # When pending prompts exist, they're shown in the queued prompts widget above the input.
    # The footer only shows the status to avoid redundancy.
    assert format_session_status(state) == "running"


def test_format_session_status_shows_paused_when_queue_is_paused() -> None:
    state = SessionState(
        running=False,
        paused=True,
        pending_prompts=("first queued prompt", "second queued prompt"),
    )

    assert format_session_status(state) == "paused"


def test_print_prompt_accepted_uses_simple_grey_background() -> None:
    with patch("coding_assistant.cli.output.rich_print") as mock_print:
        from coding_assistant.cli.output import print_active_prompt

        print_active_prompt("Do the task")

    # print_active_prompt adds: leading blank + prompt line
    assert len(mock_print.call_args_list) == 2
    assert mock_print.call_args_list[0].args == ()  # leading blank
    content = mock_print.call_args_list[1].args[0]
    assert "Do the task" in content
    assert "▌" in content


@pytest.mark.asyncio
async def test_handle_submission_enqueues_steering_prompt() -> None:
    session = Mock()
    session.enqueue_steering_prompt = AsyncMock(return_value=True)

    should_exit = await _handle_submission(
        session=session,
        content="fix this next",
        submit_type=PromptSubmitType.STEERING,
    )

    assert should_exit is False
    session.enqueue_steering_prompt.assert_awaited_once_with("fix this next")


@pytest.mark.asyncio
async def test_handle_submission_compact_enqueues_compaction_prompt() -> None:
    session = Mock()
    session.enqueue_prompt = AsyncMock(return_value=True)

    should_exit = await _handle_submission(
        session=session,
        content="/compact",
        submit_type=PromptSubmitType.QUEUED,
    )

    assert should_exit is False
    session.enqueue_prompt.assert_awaited_once_with(
        "Immediately compact our conversation so far by using the `compact_conversation` tool.",
    )
