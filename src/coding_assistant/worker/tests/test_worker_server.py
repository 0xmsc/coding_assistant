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
    Usage,
)
from coding_assistant.remote.acp import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block
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
async def test_worker_prompt_streams_update_and_emits_commit(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="hello"), AssistantMessage(content="again")]),
        commit_metadata_provider=lambda: {"title": "Worker title"},
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
            commit = parse_jsonrpc_message(await websocket.recv())
            await websocket.send(
                jsonrpc_request(4, "session/prompt", {"sessionId": session_id, "prompt": [text_block("Again")]}),
            )
            second_updates: list[dict[str, Any]] = []
            second_response: dict[str, Any] | None = None
            while second_response is None:
                payload = parse_jsonrpc_message(await websocket.recv())
                if payload.get("method") == "session/update":
                    second_updates.append(payload)
                else:
                    second_response = payload
            second_commit = parse_jsonrpc_message(await websocket.recv())

    assert any(
        update["params"]["update"]["sessionUpdate"] == "message_delta"
        and update["params"]["update"]["appendText"] == "hello"
        for update in updates
    )
    assert response["result"] == {"stopReason": "end_turn"}
    assert commit["method"] == "_session/commit"
    assert commit["params"]["sessionId"] == session_id
    assert commit["params"]["baseVersion"] == 7
    assert commit["params"]["stopReason"] == "end_turn"
    assert commit["params"]["_meta"] == {"title": "Worker title"}
    assert [message["role"] for message in commit["params"]["messages"]] == ["user", "assistant"]
    assert any(
        update["params"]["update"]["sessionUpdate"] == "message_delta"
        and update["params"]["update"]["appendText"] == "again"
        for update in second_updates
    )
    assert second_response["result"] == {"stopReason": "end_turn"}
    assert second_commit["method"] == "_session/commit"
    assert second_commit["params"]["baseVersion"] == 8
    assert second_commit["params"]["_meta"] == {"title": "Worker title"}
    assert [message["role"] for message in second_commit["params"]["messages"]] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_worker_cancel_produces_cancelled_commit(tmp_path: Path) -> None:
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
            commit = parse_jsonrpc_message(await websocket.recv())

    assert cancel_response["result"] is None
    assert prompt_response["result"] == {"stopReason": "cancelled"}
    assert commit["params"]["stopReason"] == "cancelled"


@pytest.mark.asyncio
async def test_worker_streams_tool_lifecycle_and_commits_tool_messages(tmp_path: Path) -> None:
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
            commit: dict[str, Any] | None = None
            while commit is None:
                payload = parse_jsonrpc_message(await websocket.recv())
                messages.append(payload)
                if payload.get("method") == "_session/commit":
                    commit = payload

    updates = [message for message in messages if message.get("method") == "session/update"]
    assert all(update["params"]["update"]["sessionUpdate"] in {"message_added", "message_delta"} for update in updates)
    assert commit is not None
    assert [message["role"] for message in commit["params"]["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


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
