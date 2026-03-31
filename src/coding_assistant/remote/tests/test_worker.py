from __future__ import annotations

import asyncio
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rich.markdown import Markdown
from rich.panel import Panel

from coding_assistant.app.default_agent import DefaultAgentBundle, DefaultAgentConfig
from coding_assistant.app.output import run_session_output
from coding_assistant.app.terminal_ui import TerminalUiMode
from coding_assistant.core.agent_session import (
    AgentSession,
    PromptAcceptedEvent,
    SessionState,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.core.history import build_system_prompt
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    ToolCall,
    Usage,
)
from coding_assistant.remote.server import WorkerServer
from coding_assistant.remote.worker import run_worker


class ScriptedStreamer:
    def __init__(self, script: list[AssistantMessage]) -> None:
        self.script = list(script)

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        if not self.script:
            raise AssertionError("Streamer script exhausted.")

        action = self.script.pop(0)
        if isinstance(action.content, str) and action.content:
            yield ContentDeltaEvent(content=action.content)

        yield CompletionEvent(completion=Completion(message=action, usage=Usage(tokens=10, cost=0.0)))


def make_system_history() -> list[BaseMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


def make_agent_session(*, completion_streamer: Any) -> AgentSession:
    return AgentSession(
        history=make_system_history(),
        model="test-model",
        tools=[],
        completion_streamer=completion_streamer,
    )


@pytest.mark.asyncio
async def test_run_session_output_renders_system_message_and_streamed_content() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.app.output.print_system_message") as mock_print_system,
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(run_session_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            assert await session.enqueue_prompt("Hi") is True

            while session.state.running or session.state.queued_prompt_count:
                await asyncio.sleep(0)

            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    mock_print_system.assert_called_once_with(system_message)
    panel_blocks = [
        call.args[0] for call in mock_rich_print.call_args_list if call.args and isinstance(call.args[0], Panel)
    ]
    markdown_blocks = [
        call.args[0] for call in mock_rich_print.call_args_list if call.args and isinstance(call.args[0], Markdown)
    ]
    assert len(panel_blocks) == 1
    assert isinstance(panel_blocks[0].renderable, Markdown)
    assert panel_blocks[0].renderable.markup == "Hi"
    assert [block.markup for block in markdown_blocks] == ["Hello from the worker"]


@pytest.mark.asyncio
async def test_run_session_output_prints_accepted_prompt_before_run_output() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.app.output.print_system_message"),
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(run_session_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(PromptAcceptedEvent(content="Do the task"))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    assert mock_rich_print.call_args_list[0].args == ()
    assert isinstance(mock_rich_print.call_args_list[1].args[0], Panel)
    panel = mock_rich_print.call_args_list[1].args[0]
    assert isinstance(panel.renderable, Markdown)
    assert panel.renderable.markup == "Do the task"


@pytest.mark.asyncio
async def test_run_session_output_prints_tool_calls_without_extra_spacing() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    system_message = SystemMessage(content="System")
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

    with (
        patch("coding_assistant.app.output.print_system_message"),
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(run_session_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(ContentDeltaEvent(content="Can you read README.md?"))
            session._publish_event(ToolCallsEvent(message=tool_call_message))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    assert len(mock_rich_print.call_args_list) == 4
    assert mock_rich_print.call_args_list[0].args == ()
    assert isinstance(mock_rich_print.call_args_list[1].args[0], Markdown)
    assert mock_rich_print.call_args_list[1].args[0].markup == "Can you read README.md?"
    assert mock_rich_print.call_args_list[2].args == ()
    assert mock_rich_print.call_args_list[3].args == (
        '[bold yellow]▶[/bold yellow] shell_execute(command="cat README.md")',
    )


@pytest.mark.asyncio
async def test_run_session_output_prints_status_updates_when_enabled() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.app.output.print_system_message"),
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(
            run_session_output(session=session, system_message=system_message, show_state_updates=True)
        )
        try:
            await asyncio.sleep(0)
            session._publish_event(
                StateChangedEvent(
                    state=SessionState(
                        promptable=True,
                        running=False,
                        queued_prompt_count=2,
                        pending_prompts=("queued one", "queued two"),
                    )
                )
            )
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    printed_lines = [call.args[0] for call in mock_rich_print.call_args_list if call.args]
    assert "[dim]idle | queued: 0[/dim]" in printed_lines
    assert "[dim]idle | queued: 2 | next: queued one | then: queued two[/dim]" in printed_lines


@pytest.mark.asyncio
async def test_run_worker_starts_worker_server_with_worker_tools_and_local_output(tmp_path: Path) -> None:
    args = Namespace(
        instructions=[],
        mcp_servers=[],
        model="gpt-4",
        print_mcp_tools=False,
        skills_directories=[],
        trace=False,
        wait_for_debugger=False,
        worker=True,
    )
    config = DefaultAgentConfig(working_directory=tmp_path)
    captured: dict[str, object] = {}

    @asynccontextmanager
    async def fake_create_default_agent(*, config: Any, include_worker_tools: bool = True) -> Any:
        captured["config"] = config
        captured["include_worker_tools"] = include_worker_tools
        yield DefaultAgentBundle(
            tools=[],
            instructions="Follow the repo instructions.",
            mcp_servers=[],
        )

    @asynccontextmanager
    async def fake_start_worker_server(*, session: Any) -> Any:
        captured["session"] = session
        yield WorkerServer(endpoint="ws://127.0.0.1:1234")

    async def fake_run_terminal_ui(
        *,
        session: Any,
        system_message: Any,
        mode: TerminalUiMode,
    ) -> None:
        captured["output_session"] = session
        captured["output_system_message"] = system_message
        captured["mode"] = mode
        await asyncio.Future()

    def resolved_future() -> asyncio.Future[None]:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        future.set_result(None)
        return future

    with (
        patch("coding_assistant.remote.worker.build_default_agent_config", return_value=config),
        patch("coding_assistant.remote.worker.create_default_agent", fake_create_default_agent),
        patch("coding_assistant.remote.worker.start_worker_server", fake_start_worker_server),
        patch("coding_assistant.remote.worker.run_terminal_ui", new=fake_run_terminal_ui),
        patch("coding_assistant.remote.worker.asyncio.Future", new=resolved_future),
    ):
        await run_worker(args)

    assert captured["config"] == config
    assert captured["include_worker_tools"] is True
    assert isinstance(captured["session"], AgentSession)
    assert captured["session"].history == [
        SystemMessage(content=build_system_prompt(instructions="Follow the repo instructions.")),
    ]
    assert captured["output_session"] is captured["session"]
    assert captured["mode"] is TerminalUiMode.READONLY
