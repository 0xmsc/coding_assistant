from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect

from coding_assistant.core.agent_session import AgentSession
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
from coding_assistant.remote.jsonrpc import jsonrpc_notification, jsonrpc_request, parse_jsonrpc_message, text_block
from coding_assistant.remote.protocol import messages_to_jsonrpc
from coding_assistant.worker.server import (
    WorkerRuntimeConfig,
    _execute_run,
    _WorkerConnectionState,
    start_session_worker_server,
)


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


class BlockingUpdateSocket:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.update_started = asyncio.Event()
        self.release_update = asyncio.Event()

    async def send(self, message: str) -> None:
        payload = parse_jsonrpc_message(message)
        if payload.get("method") == "session/update" and not self.update_started.is_set():
            self.update_started.set()
            await self.release_update.wait()
        self.sent.append(payload)


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


def _run_params(
    *,
    messages: list[BaseMessage] | None = None,
    prompt: str = "Do it",
) -> dict[str, Any]:
    return {
        "sessionId": "sess_test",
        "messages": messages_to_jsonrpc(messages or [SystemMessage(content="system")]),
        "prompt": [text_block(prompt)],
    }


@pytest.mark.asyncio
async def test_worker_accepts_exactly_one_run(tmp_path: Path) -> None:
    streamer = BlockingStreamer()
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=streamer)

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket, connect(server.endpoint) as second_websocket:
            await websocket.send(
                jsonrpc_request(2, "_worker/run", _run_params()),
            )
            await asyncio.wait_for(streamer.started.wait(), timeout=1)
            await second_websocket.send(jsonrpc_request(2, "_worker/run", _run_params()))
            second_run = parse_jsonrpc_message(await second_websocket.recv())
            streamer.release.set()

    assert second_run["error"]["message"] == "Worker already has a run."


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
            await websocket.send(
                jsonrpc_request(2, "_worker/run", _run_params()),
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
        "stopReason": "end_turn",
        "_meta": {"title": "Worker title"},
    }


@pytest.mark.asyncio
async def test_worker_sends_all_session_updates_before_run_result() -> None:
    streamer = ScriptedStreamer([AssistantMessage(content="hello")])
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=streamer)
    session = AgentSession(
        history=[SystemMessage(content="system")],
        model=runtime.model,
        tools=runtime.tools,
        completion_streamer=streamer,
    )
    websocket: Any = BlockingUpdateSocket()
    execute_task = asyncio.create_task(
        _execute_run(
            websocket=websocket,
            response_id=2,
            session_id="sess_test",
            prompt="Do it",
            session=session,
            state=_WorkerConnectionState(session_id="sess_test", session=session),
            runtime=runtime,
            finished=asyncio.Event(),
        ),
    )

    await asyncio.wait_for(websocket.update_started.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not any(message.get("id") == 2 for message in websocket.sent)

    websocket.release_update.set()
    await asyncio.wait_for(execute_task, timeout=1)

    result_index = next(index for index, message in enumerate(websocket.sent) if message.get("id") == 2)
    assert all(message.get("method") == "session/update" for message in websocket.sent[:result_index])
    assert any(
        message.get("params", {}).get("update", {}).get("sessionUpdate") == "message_added"
        for message in websocket.sent[:result_index]
    )


@pytest.mark.asyncio
async def test_worker_cancel_returns_cancelled_result(tmp_path: Path) -> None:
    streamer = BlockingStreamer()
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=streamer)

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            await websocket.send(
                jsonrpc_request(2, "_worker/run", _run_params()),
            )
            await asyncio.wait_for(streamer.started.wait(), timeout=1)
            await websocket.send(jsonrpc_request(3, "session/cancel", {"sessionId": "sess_test"}))
            cancel_response = parse_jsonrpc_message(await websocket.recv())
            prompt_response = parse_jsonrpc_message(await websocket.recv())

    assert cancel_response["result"] is None
    assert prompt_response["result"] == {"stopReason": "cancelled"}


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
            await websocket.send(
                jsonrpc_request(2, "_worker/run", _run_params(prompt="Use tool")),
            )

            messages: list[dict[str, Any]] = []
            response: dict[str, Any] | None = None
            while response is None:
                payload = parse_jsonrpc_message(await websocket.recv())
                messages.append(payload)
                if payload.get("id") == 2:
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
async def test_worker_requires_valid_run_params() -> None:
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=ScriptedStreamer([]))

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            invalid_params = _run_params()
            del invalid_params["prompt"]
            await websocket.send(jsonrpc_request(1, "_worker/run", invalid_params))
            invalid_run = parse_jsonrpc_message(await websocket.recv())

    assert invalid_run["error"]["message"] == "_worker/run requires a prompt array."


@pytest.mark.asyncio
async def test_worker_ignores_run_notification_without_claiming_run() -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="done")]),
    )

    async with start_session_worker_server(runtime=runtime) as server:
        async with connect(server.endpoint) as websocket:
            await websocket.send(jsonrpc_notification("_worker/run", _run_params()))
            await websocket.send(jsonrpc_request(1, "_worker/run", _run_params()))
            response: dict[str, Any] | None = None
            while response is None:
                payload = parse_jsonrpc_message(await websocket.recv())
                if payload.get("id") == 1:
                    response = payload

    assert response["result"]["stopReason"] == "end_turn"
