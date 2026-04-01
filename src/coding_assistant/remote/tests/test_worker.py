from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any, cast
from unittest.mock import patch

import pytest
from rich.markdown import Markdown
from rich.panel import Panel
from rich.styled import Styled

from coding_assistant.app.terminal_ui import run_session_output
from coding_assistant.core.agent_session import (
    AgentSession,
    PromptAcceptedEvent,
    PromptStartedEvent,
    SessionState,
    StateChangedEvent,
    ToolCallsEvent,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    StatusEvent,
    SystemMessage,
    ToolCall,
    Usage,
)
from coding_assistant.remote.protocol import (
    CommandAcceptedMessage,
    PromptCommand,
    PromptStartedMessage,
    parse_worker_message,
)
from coding_assistant.remote.server import _handle_supervisor_message, _session_event_to_message


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
        patch("coding_assistant.app.terminal_ui.print_system_message") as mock_print_system,
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
    styled_blocks = [
        call.args[0] for call in mock_rich_print.call_args_list if call.args and isinstance(call.args[0], Styled)
    ]
    markdown_blocks = [
        call.args[0] for call in mock_rich_print.call_args_list if call.args and isinstance(call.args[0], Markdown)
    ]
    assert len(panel_blocks) == 0
    assert len(styled_blocks) == 1
    assert isinstance(styled_blocks[0].renderable, Markdown)
    assert styled_blocks[0].renderable.markup == "Hi"
    assert [block.markup for block in markdown_blocks] == ["Hello from the worker"]


@pytest.mark.asyncio
async def test_run_session_output_prints_started_prompt_before_run_output() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.app.terminal_ui.print_system_message"),
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(run_session_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(PromptAcceptedEvent(content="Do the task"))
            session._publish_event(PromptStartedEvent(content="Do the task"))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    assert len(mock_rich_print.call_args_list) == 1
    styled = mock_rich_print.call_args_list[0].args[0]
    assert isinstance(styled, Styled)
    assert isinstance(styled.renderable, Markdown)
    assert styled.renderable.markup == "Do the task"


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
        patch("coding_assistant.app.terminal_ui.print_system_message"),
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
        patch("coding_assistant.app.terminal_ui.print_system_message"),
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
async def test_run_session_output_prints_status_events_as_info_lines() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.app.terminal_ui.print_system_message"),
        patch("coding_assistant.app.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(run_session_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(StatusEvent(message="Retrying LLM request"))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    printed_lines = [call.args[0] for call in mock_rich_print.call_args_list if call.args]
    assert printed_lines == ["[bold blue]i[/bold blue] Retrying LLM request"]


def test_session_event_to_message_includes_state_for_prompt_started() -> None:
    state = SessionState(
        promptable=True,
        running=True,
        queued_prompt_count=1,
        pending_prompts=("next queued prompt",),
    )

    message = _session_event_to_message(PromptStartedEvent(content="Do the task"), state=state)

    assert message == PromptStartedMessage(
        content="Do the task",
        promptable=True,
        remote_connected=True,
        running=True,
        queued_prompt_count=1,
    )


@pytest.mark.asyncio
async def test_handle_supervisor_message_replies_with_authoritative_state_snapshot() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send(self, message: str) -> None:
            self.messages.append(message)

    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    websocket = FakeWebSocket()

    try:
        await _handle_supervisor_message(
            cast(Any, websocket),
            session,
            PromptCommand(request_id="request-1", prompt="Do the task", mode="queue"),
        )
    finally:
        await session.close()

    assert len(websocket.messages) == 1
    response = parse_worker_message(websocket.messages[0])
    assert response == CommandAcceptedMessage(
        request_id="request-1",
        promptable=True,
        remote_connected=True,
        running=False,
        queued_prompt_count=1,
    )
