import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from coding_assistant.cli.ui import _handle_submission, PromptSubmitType, run_cli
from coding_assistant.cli.agent import (
    CliAgentBundle,
    build_cli_agent_config,
    create_cli_agent,
)
from coding_assistant.cli.main import main, parse_args
from coding_assistant.core.agent_session import AgentSession, RunFinishedEvent
from coding_assistant.core.compacting_session import AutoCompactingSession
from coding_assistant.core.runtime import build_initial_system_message
from coding_assistant.llm.types import AssistantMessage, SystemMessage, UserMessage
from coding_assistant.testing.fake_openai import run_fake_openai_server


def test_parse_args_valid() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.model == "gpt-4"


def test_parse_args_defaults() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4"]):
        args = parse_args()
        assert args.skills_directories == []
        assert args.bell is True
        assert args.auto_compact is True
        assert args.auto_compact_token_budget is None


def test_parse_args_no_bell() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--no-bell"]):
        args = parse_args()
        assert args.bell is False


def test_parse_args_bell() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--bell"]):
        args = parse_args()
        assert args.bell is True


def test_parse_args_auto_compact_flags() -> None:
    with patch(
        "sys.argv",
        ["coding-assistant", "--model", "gpt-4", "--no-auto-compact", "--auto-compact-token-budget", "50000"],
    ):
        args = parse_args()
        assert args.auto_compact is False
        assert args.auto_compact_token_budget == 50000


def test_parse_args_with_multiple_flags() -> None:
    with patch("sys.argv", ["coding-assistant", "--model", "gpt-4", "--trace"]):
        args = parse_args()
        assert args.trace is True


def test_build_cli_agent_config_from_args(tmp_path: Any) -> None:
    args = type("MockArgs", (), {})()
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
        bell=True,
        instructions=[],
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

    with (
        patch("coding_assistant.cli.ui.create_cli_agent", fake_create_cli_agent),
        patch("coding_assistant.cli.ui._run_ui", new=AsyncMock()) as mock_run_ui,
    ):
        await run_cli(args)

    assert mock_run_ui.await_args is not None
    session = mock_run_ui.await_args.kwargs["session"]
    assert isinstance(session, AutoCompactingSession)
    assert session.history == [
        SystemMessage(content="Follow the repo instructions."),
    ]
    assert mock_run_ui.await_args.kwargs["system_message"] == SystemMessage(
        content="Follow the repo instructions.",
    )
    assert mock_run_ui.await_args.kwargs["history_path"].name == "history"
    assert mock_run_ui.await_args.kwargs["bell"] is True


@pytest.mark.asyncio
async def test_run_cli_with_auto_compact_disabled_uses_raw_session() -> None:
    args = Namespace(
        bell=True,
        instructions=[],
        model="gpt-4",
        skills_directories=[],
        trace=False,
        wait_for_debugger=False,
        auto_compact=False,
        auto_compact_token_budget=None,
    )

    @asynccontextmanager
    async def fake_create_cli_agent(*, config: Any) -> Any:
        del config
        yield CliAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
        )

    with (
        patch("coding_assistant.cli.ui.create_cli_agent", fake_create_cli_agent),
        patch("coding_assistant.cli.ui._run_ui", new=AsyncMock()) as mock_run_ui,
    ):
        await run_cli(args)

    assert mock_run_ui.await_args is not None
    session = mock_run_ui.await_args.kwargs["session"]
    assert type(session) is AgentSession


@pytest.mark.asyncio
async def test_cli_agent_smoke_runs_against_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    args = Namespace(
        instructions=[],
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
        ),
    ]


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
