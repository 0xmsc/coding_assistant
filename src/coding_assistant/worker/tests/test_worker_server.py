from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect

from coding_assistant.llm.types import (
    AssistantMessage,
    BaseMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    TextToolResult,
    Tool,
    ToolCall,
    UserMessage,
    Usage,
)
from coding_assistant.remote.jsonrpc import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block
from coding_assistant.remote.protocol import messages_to_jsonrpc
from coding_assistant.worker.server import WorkerRuntimeConfig, start_session_worker_server


class ScriptedStreamer:
    def __init__(self, script: list[AssistantMessage]) -> None:
        self.script = list(script)

    async def __call__(
        self, messages: Any, tools: Any, model: Any
    ) -> AsyncIterator[ContentDeltaEvent | CompletionEvent]:
        del messages, tools, model
        if not self.script:
            raise AssertionError("Streamer script exhausted.")
        message = self.script.pop(0)
        if isinstance(message.content, str) and message.content:
            yield ContentDeltaEvent(content=message.content)
        yield CompletionEvent(completion=Completion(message=message, usage=Usage(tokens=1, cost=0.0)))


class BlockingStreamer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[CompletionEvent]:
        del messages, tools, model
        self.started.set()
        await self.release.wait()
        yield CompletionEvent(
            completion=Completion(message=AssistantMessage(content="done"), usage=Usage(tokens=1, cost=0.0)),
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


async def _initialize(websocket: ClientConnection) -> None:
    await websocket.send(
        jsonrpc_request(
            1,
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {},
                "clientInfo": {"name": "test", "title": "Test", "version": "0.0.0"},
            },
        ),
    )
    response = parse_jsonrpc_message(await websocket.recv())
    assert response["result"]["protocolVersion"] == ACP_PROTOCOL_VERSION


async def _start_session(
    websocket: ClientConnection,
    *,
    workspace: str,
    messages: list[BaseMessage] | None = None,
    request_id: int = 2,
) -> str:
    await websocket.send(
        jsonrpc_request(
            request_id,
            "_session/start",
            {
                "sessionId": "sess_test",
                "baseVersion": 7,
                "messages": messages_to_jsonrpc(messages or [SystemMessage(content="system")]),
                "workspace": workspace,
                "config": {"model": "test-model"},
            },
        ),
    )
    response = parse_jsonrpc_message(await websocket.recv())
    session_id = response["result"]["sessionId"]
    assert isinstance(session_id, str)
    return session_id


@pytest.mark.asyncio
async def test_worker_starts_session_from_manager_state(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=ScriptedStreamer([]))

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _start_session(websocket, workspace=str(tmp_path))

            await websocket.send(
                jsonrpc_request(
                    3,
                    "_session/start",
                    {
                        "sessionId": "sess_other",
                        "baseVersion": 0,
                        "messages": [],
                        "workspace": str(tmp_path),
                    },
                ),
            )
            second_start = parse_jsonrpc_message(await websocket.recv())

    assert session_id == "sess_test"
    assert second_start["error"]["message"] == "Worker session is already started."


@pytest.mark.asyncio
async def test_worker_prompt_streams_updates_and_returns_result(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="hello")]),
        finish_metadata_provider=lambda: {"title": "Worker title"},
    )

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _start_session(websocket, workspace=str(tmp_path))

            await websocket.send(
                jsonrpc_request(3, "session/prompt", {"sessionId": session_id, "prompt": [text_block("Do it")]}),
            )
            updates: list[dict[str, Any]] = []
            response: dict[str, Any] | None = None
            while response is None:
                payload = parse_jsonrpc_message(await websocket.recv())
                if payload.get("method") == "session/update":
                    updates.append(payload)
                else:
                    response = payload
            await server.wait_finished()

    assert any(
        update["params"]["update"]["sessionUpdate"] == "message_delta"
        and update["params"]["update"]["appendText"] == "hello"
        for update in updates
    )
    assert response["result"] == {
        "baseVersion": 7,
        "messages": messages_to_jsonrpc([UserMessage(content="Do it"), AssistantMessage(content="hello")]),
        "stopReason": "end_turn",
        "_meta": {"title": "Worker title"},
    }


@pytest.mark.asyncio
async def test_worker_cancel_returns_cancelled_result(tmp_path: Path) -> None:
    streamer = BlockingStreamer()
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=streamer)

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _start_session(websocket, workspace=str(tmp_path))

            await websocket.send(
                jsonrpc_request(3, "session/prompt", {"sessionId": session_id, "prompt": [text_block("Do it")]}),
            )
            await asyncio.wait_for(streamer.started.wait(), timeout=1)
            await websocket.send(jsonrpc_request(4, "session/cancel", {"sessionId": session_id}))
            cancel_response = parse_jsonrpc_message(await websocket.recv())
            prompt_response = parse_jsonrpc_message(await websocket.recv())

    assert cancel_response["result"] is None
    assert prompt_response["result"] == {
        "baseVersion": 7,
        "messages": messages_to_jsonrpc([UserMessage(content="Do it")]),
        "stopReason": "cancelled",
    }


@pytest.mark.asyncio
async def test_worker_streams_tool_messages_before_final_answer(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[EchoTool()],
        completion_streamer=ScriptedStreamer(
            [
                AssistantMessage(
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            function=FunctionCall(name="echo_tool", arguments='{"text": "hello"}'),
                        ),
                    ],
                ),
                AssistantMessage(content="done"),
            ],
        ),
    )

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _start_session(websocket, workspace=str(tmp_path))

            await websocket.send(
                jsonrpc_request(3, "session/prompt", {"sessionId": session_id, "prompt": [text_block("Use tool")]}),
            )

            messages: list[dict[str, Any]] = []
            response: dict[str, Any] | None = None
            while response is None:
                payload = parse_jsonrpc_message(await websocket.recv())
                messages.append(payload)
                if payload.get("id") == 3:
                    response = payload

    updates = [message["params"]["update"] for message in messages if message.get("method") == "session/update"]
    assert _normalized_updates(updates) == [
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
            "appendText": "done",
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {"role": "assistant", "content": "done"},
        },
    ]
    assert response is not None
    assert response["result"]["stopReason"] == "end_turn"


@pytest.mark.asyncio
async def test_worker_rejects_prompt_before_start_and_wrong_session_id(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=ScriptedStreamer([]))

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(
                jsonrpc_request(2, "session/prompt", {"sessionId": "sess_test", "prompt": [text_block("bad")]}),
            )
            before_start = parse_jsonrpc_message(await websocket.recv())
            await _start_session(websocket, workspace=str(tmp_path), request_id=3)
            await websocket.send(
                jsonrpc_request(4, "session/prompt", {"sessionId": "sess_other", "prompt": [text_block("bad")]}),
            )
            wrong_session = parse_jsonrpc_message(await websocket.recv())

    assert before_start["error"]["message"] == "_session/start must be called first."
    assert wrong_session["error"]["message"] == "Unknown sessionId."
