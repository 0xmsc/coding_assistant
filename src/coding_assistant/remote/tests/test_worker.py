from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from rich.markdown import Markdown
from rich.panel import Panel
from rich.styled import Styled
from websockets.asyncio.client import ClientConnection, connect

from coding_assistant.app.terminal_ui import run_session_output
from coding_assistant.core.agent_session import (
    AgentSession,
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
    Tool,
    ToolCall,
    Usage,
)
from coding_assistant.remote.acp import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block
from coding_assistant.remote.server import start_worker_server


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


class BlockingStreamer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        self.started.set()
        await self.release.wait()
        yield CompletionEvent(
            completion=Completion(
                message=AssistantMessage(content="Finished"),
                usage=Usage(tokens=10, cost=0.0),
            )
        )


class EchoTool(Tool):
    def name(self) -> str:
        return "echo_tool"

    def description(self) -> str:
        return "Echo the provided text."

    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, parameters: dict[str, Any]) -> str:
        return f"echo:{parameters['text']}"


def make_system_history() -> list[BaseMessage]:
    return [SystemMessage(content="# Instructions\n\nTest instructions")]


def make_agent_session(*, completion_streamer: Any, tools: list[Tool] | None = None) -> AgentSession:
    return AgentSession(
        history=make_system_history(),
        model="test-model",
        tools=tools or [],
        completion_streamer=completion_streamer,
    )


async def _open_acp_session(websocket: ClientConnection) -> str:
    await websocket.send(
        jsonrpc_request(
            1,
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "title": "Test Client",
                    "version": "0.0.0",
                },
            },
        )
    )
    initialize_response = parse_jsonrpc_message(await websocket.recv())
    assert initialize_response["result"]["protocolVersion"] == ACP_PROTOCOL_VERSION

    await websocket.send(
        jsonrpc_request(
            2,
            "session/new",
            {
                "cwd": "/tmp",
                "mcpServers": [],
            },
        )
    )
    session_response = parse_jsonrpc_message(await websocket.recv())
    session_id = session_response["result"]["sessionId"]
    assert isinstance(session_id, str)
    return session_id


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
    # When pending prompts exist, they're shown in the queued prompts widget, not in the footer.
    assert "[dim]idle[/dim]" in printed_lines


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


@pytest.mark.asyncio
async def test_worker_server_completes_acp_prompt_turn() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )

    try:
        async with start_worker_server(session=session) as worker_server:
            async with connect(worker_server.endpoint) as websocket:
                session_id = await _open_acp_session(websocket)

                await websocket.send(
                    jsonrpc_request(
                        3,
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [text_block("Do the task")],
                        },
                    )
                )

                updates: list[dict[str, Any]] = []
                response: dict[str, Any] | None = None
                while response is None:
                    payload = parse_jsonrpc_message(await websocket.recv())
                    if payload.get("method") == "session/update":
                        updates.append(payload)
                    else:
                        response = payload

        assert response == {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"stopReason": "end_turn"},
        }
        assert any(
            update["params"]["update"]["sessionUpdate"] == "agent_message_chunk"
            and update["params"]["update"]["content"]["text"] == "Hello from the worker"
            for update in updates
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_worker_server_rejects_busy_acp_prompt_turn() -> None:
    streamer = BlockingStreamer()
    session = make_agent_session(completion_streamer=streamer)

    try:
        assert await session.enqueue_prompt("Already busy") is True
        await asyncio.wait_for(streamer.started.wait(), timeout=1)

        async with start_worker_server(session=session) as worker_server:
            async with connect(worker_server.endpoint) as websocket:
                session_id = await _open_acp_session(websocket)
                await websocket.send(
                    jsonrpc_request(
                        3,
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [text_block("Do the task")],
                        },
                    )
                )
                response = parse_jsonrpc_message(await websocket.recv())

        assert response["id"] == 3
        assert response["error"]["message"] == "Session is busy. Wait for the current turn or cancel it."
    finally:
        streamer.release.set()
        await session.close()


@pytest.mark.asyncio
async def test_worker_server_reports_tool_call_lifecycle_over_acp() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer(
            [
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(
                                name="echo_tool",
                                arguments='{"text": "hello"}',
                            ),
                        )
                    ]
                ),
                AssistantMessage(content="Done"),
            ]
        ),
        tools=[EchoTool()],
    )

    try:
        async with start_worker_server(session=session) as worker_server:
            async with connect(worker_server.endpoint) as websocket:
                session_id = await _open_acp_session(websocket)

                await websocket.send(
                    jsonrpc_request(
                        3,
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [text_block("Use the tool")],
                        },
                    )
                )

                updates: list[dict[str, Any]] = []
                response: dict[str, Any] | None = None
                while response is None:
                    payload = parse_jsonrpc_message(await websocket.recv())
                    if payload.get("method") == "session/update":
                        updates.append(payload)
                    else:
                        response = payload

        assert response == {
            "jsonrpc": "2.0",
            "id": 3,
            "result": {"stopReason": "end_turn"},
        }
        assert any(
            update["params"]["update"]
            == {
                "sessionUpdate": "tool_call",
                "toolCallId": "call-1",
                "title": "echo_tool",
                "kind": "other",
                "status": "pending",
                "rawInput": {"text": "hello"},
            }
            for update in updates
        )
        assert any(
            update["params"]["update"]["sessionUpdate"] == "tool_call_update"
            and update["params"]["update"]["toolCallId"] == "call-1"
            and update["params"]["update"]["status"] == "in_progress"
            for update in updates
        )
        assert any(
            update["params"]["update"]["sessionUpdate"] == "tool_call_update"
            and update["params"]["update"]["toolCallId"] == "call-1"
            and update["params"]["update"]["status"] == "completed"
            and update["params"]["update"]["rawOutput"] == "echo:hello"
            and update["params"]["update"]["content"][0]["content"]["text"] == "echo:hello"
            for update in updates
        )
    finally:
        await session.close()
