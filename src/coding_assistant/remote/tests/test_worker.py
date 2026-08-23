from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest
from rich.markdown import Markdown
from websockets.asyncio.client import ClientConnection, connect

from coding_assistant.cli.ui import _run_output
from coding_assistant.core.agent_session import (
    AgentSession,
    AssistantMessageCompletedEvent,
    AssistantMessageDeltaEvent,
    PromptStartedEvent,
)
from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    ModelRetryEvent,
    ReasoningDeltaEvent,
    StatusEvent,
    SystemMessage,
    TextToolResult,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
    Usage,
)
from coding_assistant.remote.jsonrpc import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block
from coding_assistant.remote.protocol import messages_to_jsonrpc
from coding_assistant.remote.control import RemoteAgentController, RemoteAgentInfo
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
            ),
        )


class BlockingUpdateSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.update_started = asyncio.Event()
        self.release_update = asyncio.Event()
        self.prompt_response_sent = asyncio.Event()

    async def send(self, message: str) -> None:
        payload = parse_jsonrpc_message(message)
        if payload.get("method") == "session/update" and not self.update_started.is_set():
            self.update_started.set()
            await self.release_update.wait()
        self.sent.append(payload)
        if payload.get("id") == 3:
            self.prompt_response_sent.set()


class RetryStreamer:
    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[object]:
        del messages, tools, model
        yield ContentDeltaEvent(content="abandoned")
        yield ModelRetryEvent()
        yield ContentDeltaEvent(content="kept")
        yield CompletionEvent(
            completion=Completion(message=AssistantMessage(content="kept"), usage=Usage(tokens=1, cost=0.0)),
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

    async def execute(self, parameters: dict[str, Any]) -> TextToolResult:
        return TextToolResult(content=f"echo:{parameters['text']}")


def _message_payload(update: dict[str, Any]) -> dict[str, Any]:
    message = update["message"]
    assert isinstance(message, dict)
    payload = dict(message)
    payload.pop("id", None)
    payload.pop("createdAt", None)
    return payload


def _normalized_updates(updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_message_ids = [
        message_id
        for update in updates
        if update.get("sessionUpdate") == "message_added"
        and isinstance((message := update.get("message")), dict)
        and isinstance((message_id := message.get("id")), str)
    ]
    message_ids = {message_id: f"message-{index}" for index, message_id in enumerate(raw_message_ids)}
    result: list[dict[str, Any]] = []
    for update in updates:
        update_type = update.get("sessionUpdate")
        if update_type == "message_added":
            message = update["message"]
            assert isinstance(message, dict)
            message_id = message.get("id")
            assert isinstance(message_id, str)
            result.append(
                {
                    "sessionUpdate": "message_added",
                    "messageId": message_ids[message_id],
                    "message": _message_payload(update),
                }
            )
            continue
        if update_type == "message_delta":
            message_id = update["messageId"]
            assert isinstance(message_id, str)
            result.append(
                {
                    "sessionUpdate": "message_delta",
                    "messageId": message_ids.get(message_id, message_id),
                    "appendText": update["appendText"],
                }
            )
            continue
        result.append(dict(update))
    return result


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
        ),
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
        ),
    )
    session_response = parse_jsonrpc_message(await websocket.recv())
    session_id = session_response["result"]["sessionId"]
    assert isinstance(session_id, str)
    return session_id


@pytest.mark.asyncio
async def test_worker_keeps_connection_open_after_invalid_prompt() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Done")]),
    )
    try:
        async with start_worker_server(session=session) as worker_server:
            async with connect(worker_server.endpoint) as websocket:
                session_id = await _open_acp_session(websocket)
                await websocket.send(jsonrpc_request(3, "session/prompt", {"prompt": [text_block("Missing id")]}))
                invalid = parse_jsonrpc_message(await websocket.recv())

                await websocket.send(
                    jsonrpc_request(
                        4,
                        "session/prompt",
                        {
                            "sessionId": session_id,
                            "prompt": [text_block("Valid")],
                        },
                    ),
                )
                response: dict[str, Any] | None = None
                while response is None:
                    payload = parse_jsonrpc_message(await websocket.recv())
                    if payload.get("id") == 4:
                        response = payload

        assert invalid["error"]["code"] == -32602
        assert response["result"] == {
            "messages": messages_to_jsonrpc([UserMessage(content="Valid"), AssistantMessage(content="Done")]),
            "stopReason": "end_turn",
        }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test__run_output_renders_system_message_and_streamed_content() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )
    system_message = SystemMessage(content="# Instructions\n\nTest instructions")

    with (
        patch("coding_assistant.cli.ui.print_system_message") as mock_print_system,
        patch("coding_assistant.cli.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(_run_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            assert await session.enqueue_prompt("Hi") is not None

            while session.state.running or session.state.pending_prompts:
                await asyncio.sleep(0)

            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    mock_print_system.assert_called_once_with(system_message)
    markdown_blocks = [
        call.args[0] for call in mock_rich_print.call_args_list if call.args and isinstance(call.args[0], Markdown)
    ]
    # Should have rendered the streamed content as markdown
    assert [block.markup for block in markdown_blocks] == ["Hello from the worker"]


@pytest.mark.asyncio
async def test__run_output_prints_started_prompt_before_run_output() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.cli.ui.print_system_message"),
        patch("coding_assistant.cli.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(_run_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(PromptStartedEvent(content="Do the task"))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    # print_active_prompt adds: leading blank + prompt line
    assert len(mock_rich_print.call_args_list) == 2
    assert mock_rich_print.call_args_list[0].args == ()  # leading blank
    content = mock_rich_print.call_args_list[1].args[0]
    assert "Do the task" in content


@pytest.mark.asyncio
async def test__run_output_prints_tool_calls_without_extra_spacing() -> None:
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
            ),
        ],
    )

    with (
        patch("coding_assistant.cli.ui.print_system_message"),
        patch("coding_assistant.cli.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(_run_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(
                AssistantMessageDeltaEvent(message_id="message-1", content="Can you read README.md?"),
            )
            session._publish_event(
                AssistantMessageCompletedEvent(
                    message_id="message-1",
                    completion=Completion(message=tool_call_message),
                ),
            )
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    # Each element owns its own spacing:
    # 1. blank line before content (from _print_markdown)
    # 2. markdown content
    # 3. blank line before tool call (from print_tool_calls)
    # 4. tool call line
    assert len(mock_rich_print.call_args_list) == 4
    assert mock_rich_print.call_args_list[0].args == ()
    assert isinstance(mock_rich_print.call_args_list[1].args[0], Markdown)
    assert mock_rich_print.call_args_list[1].args[0].markup == "Can you read README.md?"
    assert mock_rich_print.call_args_list[2].args == ()  # blank line before tool call
    assert "▶" in str(mock_rich_print.call_args_list[3])
    assert "shell_execute" in str(mock_rich_print.call_args_list[3])


@pytest.mark.asyncio
async def test__run_output_prints_status_events_as_info_lines() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.cli.ui.print_system_message"),
        patch("coding_assistant.cli.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(_run_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(StatusEvent(message="Retrying LLM request"))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    printed_lines = [call.args[0] for call in mock_rich_print.call_args_list if call.args]
    assert printed_lines == ["[bold blue]ℹ[/bold blue] Retrying LLM request"]


@pytest.mark.asyncio
async def test__run_output_prints_reasoning_deltas_before_content() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="unused")]),
    )
    system_message = SystemMessage(content="System")

    with (
        patch("coding_assistant.cli.ui.print_system_message"),
        patch("coding_assistant.cli.output.rich_print") as mock_rich_print,
    ):
        task = asyncio.create_task(_run_output(session=session, system_message=system_message))
        try:
            await asyncio.sleep(0)
            session._publish_event(ReasoningDeltaEvent(content="Thinking"))
            session._publish_event(AssistantMessageDeltaEvent(message_id="message-1", content="Answer"))
            await asyncio.sleep(0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await session.close()

    markdown_calls = [
        call for call in mock_rich_print.call_args_list if call.args and isinstance(call.args[0], Markdown)
    ]
    assert [call.args[0].markup for call in markdown_calls] == ["Thinking", "Answer"]
    assert markdown_calls[0].kwargs == {}
    assert markdown_calls[0].args[0].style == "dim"
    assert markdown_calls[1].kwargs == {}


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
                    ),
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
            "result": {
                "messages": messages_to_jsonrpc(
                    [UserMessage(content="Do the task"), AssistantMessage(content="Hello from the worker")]
                ),
                "stopReason": "end_turn",
            },
        }
        assert any(
            update["params"]["update"]["sessionUpdate"] == "message_delta"
            and update["params"]["update"]["appendText"] == "Hello from the worker"
            for update in updates
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_live_remote_sends_all_session_updates_before_prompt_response() -> None:
    session = make_agent_session(
        completion_streamer=ScriptedStreamer([AssistantMessage(content="Hello from the worker")]),
    )
    websocket: Any = BlockingUpdateSocket()
    controller = RemoteAgentController(
        agent_info=RemoteAgentInfo(name="test", title="Test"),
        session_id="sess_test",
        session=session,
    )
    publisher_task = asyncio.create_task(controller.publish_session_events(websocket=websocket))
    try:
        await controller.handle_jsonrpc_message(
            websocket=websocket,
            payload={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}},
        )
        await controller.handle_jsonrpc_message(
            websocket=websocket,
            payload={"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}},
        )
        await controller.handle_jsonrpc_message(
            websocket=websocket,
            payload={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": "sess_test", "prompt": [text_block("Do the task")]},
            },
        )

        await asyncio.wait_for(websocket.update_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not websocket.prompt_response_sent.is_set()

        websocket.release_update.set()
        await asyncio.wait_for(websocket.prompt_response_sent.wait(), timeout=1)

        result_index = next(index for index, message in enumerate(websocket.sent) if message.get("id") == 3)
        assert any(
            message.get("params", {}).get("update", {}).get("sessionUpdate") == "message_added"
            for message in websocket.sent[:result_index]
        )
    finally:
        publisher_task.cancel()
        await asyncio.gather(publisher_task, return_exceptions=True)
        await controller.close()
        await session.close()


@pytest.mark.asyncio
async def test_worker_server_rejects_busy_acp_prompt_turn() -> None:
    streamer = BlockingStreamer()
    session = make_agent_session(completion_streamer=streamer)

    try:
        assert await session.enqueue_prompt("Already busy") is not None
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
                    ),
                )
                response = parse_jsonrpc_message(await websocket.recv())

        assert response["id"] == 3
        assert response["error"]["message"] == "Session is busy. Wait for the current turn or cancel it."
    finally:
        streamer.release.set()
        await session.close()


@pytest.mark.asyncio
async def test_worker_uses_a_new_message_id_for_a_retried_stream() -> None:
    session = make_agent_session(completion_streamer=RetryStreamer())
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
                            "prompt": [text_block("Retry")],
                        },
                    ),
                )

                updates: list[dict[str, Any]] = []
                response: dict[str, Any] | None = None
                while response is None:
                    payload = parse_jsonrpc_message(await websocket.recv())
                    if payload.get("method") == "session/update":
                        updates.append(payload["params"]["update"])
                    else:
                        response = payload

        message_updates = [
            update for update in updates if update.get("sessionUpdate") in {"message_delta", "message_added"}
        ]
        first_delta, second_delta, completed = message_updates
        first_id = first_delta["messageId"]
        second_id = second_delta["messageId"]
        assert first_delta["appendText"] == "abandoned"
        assert second_delta["appendText"] == "kept"
        assert first_id != second_id
        assert completed["message"]["id"] == second_id
        assert completed["message"]["content"] == "kept"
        assert response["result"] == {
            "messages": messages_to_jsonrpc([UserMessage(content="Retry"), AssistantMessage(content="kept")]),
            "stopReason": "end_turn",
        }
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_worker_server_streams_tool_messages_before_final_answer() -> None:
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
                        ),
                    ],
                ),
                AssistantMessage(content="Done"),
            ],
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
                    ),
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
            "result": {
                "messages": messages_to_jsonrpc(
                    [
                        UserMessage(content="Use the tool"),
                        AssistantMessage(
                            tool_calls=[
                                ToolCall(
                                    id="call-1",
                                    function=FunctionCall(name="echo_tool", arguments='{"text": "hello"}'),
                                )
                            ]
                        ),
                        ToolMessage(tool_call_id="call-1", name="echo_tool", content="echo:hello"),
                        AssistantMessage(content="Done"),
                    ]
                ),
                "stopReason": "end_turn",
            },
        }
        update_payloads = [update["params"]["update"] for update in updates]
        assert _normalized_updates(update_payloads) == [
            {
                "sessionUpdate": "message_added",
                "messageId": "message-0",
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "echo_tool", "arguments": '{"text": "hello"}'},
                            "type": "function",
                        }
                    ],
                },
            },
            {
                "sessionUpdate": "message_added",
                "messageId": "message-1",
                "message": {
                    "role": "tool",
                    "content": "echo:hello",
                    "name": "echo_tool",
                    "tool_call_id": "call-1",
                },
            },
            {
                "sessionUpdate": "message_delta",
                "messageId": "message-2",
                "appendText": "Done",
            },
            {
                "sessionUpdate": "message_added",
                "messageId": "message-2",
                "message": {"role": "assistant", "content": "Done"},
            },
        ]
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_worker_server_streams_history_reset_on_compaction() -> None:
    compact_call = ToolCall(
        id="call-compact",
        function=FunctionCall(
            name="compact_conversation",
            arguments='{"summary": "Summary of conversation"}',
        ),
    )
    session = make_agent_session(
        completion_streamer=ScriptedStreamer(
            [
                AssistantMessage(tool_calls=[compact_call]),
                AssistantMessage(content="After compaction"),
            ],
        ),
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
                            "prompt": [text_block("Please compact")],
                        },
                    ),
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
            "result": {
                "messages": messages_to_jsonrpc(
                    [
                        UserMessage(
                            content="A summary of your conversation until now:\n\nSummary of conversation\n\nPlease continue your work.",
                        ),
                        AssistantMessage(content="After compaction"),
                    ],
                ),
                "stopReason": "end_turn",
            },
        }
        update_payloads = [update["params"]["update"] for update in updates]
        assert any(payload.get("sessionUpdate") == "history_reset" for payload in update_payloads)
        reset_index = next(
            i for i, payload in enumerate(update_payloads) if payload.get("sessionUpdate") == "history_reset"
        )
        post_reset_updates = update_payloads[reset_index:]
        assert post_reset_updates[0] == {"sessionUpdate": "history_reset"}
        assert post_reset_updates[1]["sessionUpdate"] == "message_added"
        assert post_reset_updates[1]["message"]["role"] == "system"
        assert post_reset_updates[2]["sessionUpdate"] == "message_added"
        assert post_reset_updates[2]["message"]["role"] == "user"
        assert "Summary of conversation" in post_reset_updates[2]["message"]["content"]
    finally:
        await session.close()
