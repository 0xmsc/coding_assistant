from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect

from coding_assistant.core.session_updates import (
    MessageAddedUpdate,
    MessageDeltaUpdate,
    RunUpdatedUpdate,
    SessionUpdate,
    SessionUpdatedUpdate,
)
from coding_assistant.llm.openai import ProviderModel
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    FunctionCall,
    SystemMessage,
    TextToolResult,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
    Usage,
)
from coding_assistant.manager.remote_worker import RemoteWorkerRunner
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import ManagerService
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.tests.store_helpers import create_session_store
from coding_assistant.remote.jsonrpc import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block
from coding_assistant.testing.fake_openai import run_fake_openai_server
from coding_assistant.worker.server import WorkerRuntimeConfig, start_session_worker_server


class ScriptedStreamer:
    def __init__(self, script: list[AssistantMessage]) -> None:
        self.script = list(script)

    async def __call__(
        self,
        messages: Any,
        tools: Any,
        model: Any,
    ) -> AsyncIterator[ContentDeltaEvent | CompletionEvent]:
        del messages, tools, model
        if not self.script:
            raise AssertionError("Streamer script exhausted.")
        message = self.script.pop(0)
        if isinstance(message.content, str) and message.content:
            yield ContentDeltaEvent(content=message.content)
        yield CompletionEvent(completion=Completion(message=message, usage=Usage(tokens=1, cost=0.0)))


class FinalMessageDiffersFromDeltaStreamer:
    async def __call__(
        self,
        messages: Any,
        tools: Any,
        model: Any,
    ) -> AsyncIterator[ContentDeltaEvent | CompletionEvent]:
        del messages, tools, model
        yield ContentDeltaEvent(content="draft answer")
        yield CompletionEvent(
            completion=Completion(message=AssistantMessage(content="final answer"), usage=Usage(tokens=1, cost=0.0)),
        )


class BlockingStreamer:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def __call__(self, messages: Any, tools: Any, model: Any) -> AsyncIterator[CompletionEvent]:
        del messages, tools, model
        self.started.set()
        await asyncio.Event().wait()
        yield CompletionEvent(
            completion=Completion(message=AssistantMessage(content="unused"), usage=Usage(tokens=1, cost=0.0)),
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


def _scope_params(session_id: str, prompt: str) -> dict[str, Any]:
    return {
        "_meta": {"scopeId": "scope-a"},
        "sessionId": session_id,
        "prompt": [text_block(prompt)],
    }


def _update(message: dict[str, Any]) -> dict[str, Any]:
    update = message["params"]["update"]
    assert isinstance(update, dict)
    return update


def _message_payload(update: dict[str, Any]) -> dict[str, Any]:
    payload = update["message"]
    assert isinstance(payload, dict)
    payload = dict(payload)
    payload.pop("id", None)
    payload.pop("createdAt", None)
    if payload.get("role") == "system":
        return {"role": "system"}
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


def _message_id(update: SessionUpdate) -> str:
    assert isinstance(update, MessageAddedUpdate)
    return update.message_id


def _session_payload(update: SessionUpdate) -> dict[str, Any]:
    assert isinstance(update, SessionUpdatedUpdate)
    return update.session


def _run_payload(update: SessionUpdate) -> dict[str, Any]:
    assert isinstance(update, RunUpdatedUpdate)
    return update.run


async def _recv_response(websocket: Any) -> dict[str, Any]:
    while True:
        message = parse_jsonrpc_message(await websocket.recv())
        if message.get("method") != "session/update":
            return message


async def _recv_response_with_updates(websocket: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []
    while True:
        message = parse_jsonrpc_message(await websocket.recv())
        if message.get("method") == "session/update":
            updates.append(message)
            continue
        return message, updates


async def _test_model_lister() -> list[ProviderModel]:
    return [ProviderModel(id="test-model")]


def _manager_service(*, tmp_path: Path, endpoint: str) -> tuple[ManagerService, SessionStore]:
    store = create_session_store(tmp_path)
    return ManagerService(
        store=store,
        worker_runner=RemoteWorkerRunner(endpoint=endpoint),
        model_lister=_test_model_lister,
    ), store


async def _ignore_update(update: SessionUpdate) -> None:
    del update


@pytest.mark.asyncio
async def test_remote_worker_prompt_streams_update_and_persists_to_sqlite(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="hello")]),
        finish_metadata_provider=lambda: {"title": "Remote worker title"},
    )

    async with start_session_worker_server(runtime=runtime) as worker_server:
        service, store = _manager_service(tmp_path=tmp_path, endpoint=worker_server.endpoint)
        created = store.create_session(
            scope_id="scope-a",
            messages=[SystemMessage(content="system")],
            metadata={"model": "test-model"},
        )
        updates: list[SessionUpdate] = []

        async def collect_update(update: SessionUpdate) -> None:
            updates.append(update)

        result = await service.prompt(
            params=_scope_params(created.record.session_id, "Do it"),
            on_update=collect_update,
        )
        loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert result.stop_reason == "end_turn"
    session_update = _session_payload(updates[-1])
    run_updates = [_run_payload(update) for update in updates if isinstance(update, RunUpdatedUpdate)]
    message_updates = [update for update in updates if not isinstance(update, (RunUpdatedUpdate, SessionUpdatedUpdate))]
    assert [update["status"] for update in run_updates] == ["running", "completed"]
    assert [update.message for update in message_updates if isinstance(update, MessageAddedUpdate)] == [
        UserMessage(content="Do it"),
        AssistantMessage(content="hello"),
    ]
    assert message_updates[1] == MessageDeltaUpdate(message_id=_message_id(message_updates[2]), append_text="hello")
    assert session_update["title"] == "Remote worker title"
    assert loaded.record.title == "Remote worker title"
    assert loaded.messages == [
        SystemMessage(content="system"),
        UserMessage(content="Do it"),
        AssistantMessage(content="hello"),
    ]


@pytest.mark.asyncio
async def test_remote_worker_title_preserves_model_for_second_prompt(tmp_path: Path) -> None:
    first_runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="first")]),
        finish_metadata_provider=lambda: {"title": "Remote worker title"},
    )
    store = create_session_store(tmp_path)
    created = store.create_session(
        scope_id="scope-a",
        messages=[SystemMessage(content="system")],
        metadata={"model": "test-model"},
    )

    async with start_session_worker_server(runtime=first_runtime) as worker_server:
        service = ManagerService(
            store=store,
            worker_runner=RemoteWorkerRunner(endpoint=worker_server.endpoint),
            model_lister=_test_model_lister,
        )
        await service.prompt(
            params=_scope_params(created.record.session_id, "First prompt"),
            on_update=_ignore_update,
        )
    after_first_prompt = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    second_runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="second")]),
    )
    async with start_session_worker_server(runtime=second_runtime) as worker_server:
        service = ManagerService(
            store=store,
            worker_runner=RemoteWorkerRunner(endpoint=worker_server.endpoint),
            model_lister=_test_model_lister,
        )
        second_result = await service.prompt(
            params=_scope_params(created.record.session_id, "Second prompt"),
            on_update=_ignore_update,
        )
    loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert after_first_prompt.record.title == "Remote worker title"
    assert after_first_prompt.record.metadata == {"model": "test-model"}
    assert second_result.stop_reason == "end_turn"
    assert loaded.record.metadata == {"model": "test-model"}
    assert loaded.messages[-2:] == [
        UserMessage(content="Second prompt"),
        AssistantMessage(content="second"),
    ]


@pytest.mark.asyncio
async def test_remote_worker_persists_final_message_not_streamed_delta(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=FinalMessageDiffersFromDeltaStreamer(),
    )

    async with start_session_worker_server(runtime=runtime) as worker_server:
        service, store = _manager_service(tmp_path=tmp_path, endpoint=worker_server.endpoint)
        created = store.create_session(
            scope_id="scope-a",
            messages=[SystemMessage(content="system")],
            metadata={"model": "test-model"},
        )
        updates: list[SessionUpdate] = []

        async def collect_update(update: SessionUpdate) -> None:
            updates.append(update)

        result = await service.prompt(
            params=_scope_params(created.record.session_id, "Do it"),
            on_update=collect_update,
        )
        loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    message_updates = [update for update in updates if not isinstance(update, (RunUpdatedUpdate, SessionUpdatedUpdate))]
    assert result.stop_reason == "end_turn"
    assert message_updates[1] == MessageDeltaUpdate(
        message_id=_message_id(message_updates[2]), append_text="draft answer"
    )
    assert message_updates[2] == MessageAddedUpdate(
        message_id=_message_id(message_updates[2]),
        message=AssistantMessage(content="final answer"),
    )
    assert loaded.messages == [
        SystemMessage(content="system"),
        UserMessage(content="Do it"),
        AssistantMessage(content="final answer"),
    ]


@pytest.mark.asyncio
async def test_remote_worker_two_sequential_prompts_advance_history_once(tmp_path: Path) -> None:
    first_runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="first answer")]),
    )
    second_runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="second answer")]),
    )

    async with start_session_worker_server(runtime=first_runtime) as first_worker:
        service, store = _manager_service(tmp_path=tmp_path, endpoint=first_worker.endpoint)
        created = store.create_session(
            scope_id="scope-a",
            messages=[SystemMessage(content="system")],
            metadata={"model": "test-model"},
        )

        first = await service.prompt(
            params=_scope_params(created.record.session_id, "first"),
            on_update=lambda update: _ignore_update(update),
        )
        await first_worker.wait_finished()

    async with start_session_worker_server(runtime=second_runtime) as second_worker:
        service, store = _manager_service(tmp_path=tmp_path, endpoint=second_worker.endpoint)
        second = await service.prompt(
            params=_scope_params(created.record.session_id, "second"),
            on_update=lambda update: _ignore_update(update),
        )
        await second_worker.wait_finished()
        loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert first.stop_reason == "end_turn"
    assert second.stop_reason == "end_turn"
    assert [message.role for message in loaded.messages] == ["system", "user", "assistant", "user", "assistant"]
    assert [getattr(message, "content", None) for message in loaded.messages] == [
        "system",
        "first",
        "first answer",
        "second",
        "second answer",
    ]


@pytest.mark.asyncio
async def test_remote_worker_streams_tool_messages_before_final_answer_without_duplicates(
    tmp_path: Path,
) -> None:
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="echo_tool", arguments='{"text": "hello"}'),
    )
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[EchoTool()],
        completion_streamer=ScriptedStreamer(
            [
                AssistantMessage(tool_calls=[tool_call]),
                AssistantMessage(content="done"),
            ],
        ),
    )

    async with start_session_worker_server(runtime=runtime) as worker_server:
        service, store = _manager_service(tmp_path=tmp_path, endpoint=worker_server.endpoint)
        created = store.create_session(
            scope_id="scope-a",
            messages=[SystemMessage(content="system")],
            metadata={"model": "test-model"},
        )
        updates: list[SessionUpdate] = []

        async def collect_update(update: SessionUpdate) -> None:
            updates.append(update)

        result = await service.prompt(
            params=_scope_params(created.record.session_id, "Use tool"),
            on_update=collect_update,
        )
        loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert result.stop_reason == "end_turn"
    run_updates = [_run_payload(update) for update in updates if isinstance(update, RunUpdatedUpdate)]
    message_updates = [update for update in updates if not isinstance(update, (RunUpdatedUpdate, SessionUpdatedUpdate))]
    assert [update["status"] for update in run_updates] == ["running", "completed"]
    assert [update.message for update in message_updates if isinstance(update, MessageAddedUpdate)] == [
        UserMessage(content="Use tool"),
        AssistantMessage(tool_calls=[tool_call]),
        ToolMessage(tool_call_id="call-1", name="echo_tool", content="echo:hello"),
        AssistantMessage(content="done"),
    ]
    assert message_updates[3] == MessageDeltaUpdate(message_id=_message_id(message_updates[4]), append_text="done")
    assert loaded.messages == [
        SystemMessage(content="system"),
        UserMessage(content="Use tool"),
        AssistantMessage(tool_calls=[tool_call]),
        ToolMessage(tool_call_id="call-1", name="echo_tool", content="echo:hello"),
        AssistantMessage(content="done"),
    ]


@pytest.mark.asyncio
async def test_remote_worker_cancel_reaches_active_worker(tmp_path: Path) -> None:
    streamer = BlockingStreamer()
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=streamer)

    async with start_session_worker_server(runtime=runtime) as worker_server:
        service, store = _manager_service(tmp_path=tmp_path, endpoint=worker_server.endpoint)
        created = store.create_session(
            scope_id="scope-a",
            messages=[SystemMessage(content="system")],
            metadata={"model": "test-model"},
        )
        prompt_task = asyncio.create_task(
            service.prompt(
                params=_scope_params(created.record.session_id, "cancel me"),
                on_update=lambda update: _ignore_update(update),
            ),
        )
        await asyncio.wait_for(streamer.started.wait(), timeout=1)

        await service.cancel(params={"_meta": {"scopeId": "scope-a"}, "sessionId": created.record.session_id})
        result = await asyncio.wait_for(prompt_task, timeout=1)
        loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert result.stop_reason == "cancelled"
    assert [message.role for message in loaded.messages] == ["system", "user"]


@pytest.mark.asyncio
async def test_remote_worker_cancel_during_manager_update_reaches_future_worker(tmp_path: Path) -> None:
    streamer = ScriptedStreamer([AssistantMessage(content="should not run")])
    runtime = WorkerRuntimeConfig(model="test-model", tools=[], completion_streamer=streamer)
    running_update_started = asyncio.Event()
    release_running_update = asyncio.Event()

    async def block_running_update(update: SessionUpdate) -> None:
        if isinstance(update, RunUpdatedUpdate) and update.run.get("status") == "running":
            running_update_started.set()
            await release_running_update.wait()

    async with start_session_worker_server(runtime=runtime) as worker_server:
        service, store = _manager_service(tmp_path=tmp_path, endpoint=worker_server.endpoint)
        created = store.create_session(
            scope_id="scope-a",
            messages=[SystemMessage(content="system")],
            metadata={"model": "test-model"},
        )
        prompt_task = asyncio.create_task(
            service.prompt(
                params=_scope_params(created.record.session_id, "cancel before worker starts"),
                on_update=block_running_update,
            ),
        )
        await asyncio.wait_for(running_update_started.wait(), timeout=1)

        await service.cancel(params={"_meta": {"scopeId": "scope-a"}, "sessionId": created.record.session_id})
        release_running_update.set()
        result = await asyncio.wait_for(prompt_task, timeout=1)
        loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert result.stop_reason == "cancelled"
    assert [message.role for message in loaded.messages] == ["system", "user"]
    assert len(streamer.script) == 1


@pytest.mark.asyncio
async def test_manager_server_uses_remote_worker_and_replays_persisted_history(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="server answer")]),
    )

    async with start_session_worker_server(runtime=runtime) as worker_server:
        service, _store = _manager_service(tmp_path=tmp_path, endpoint=worker_server.endpoint)
        async with start_manager_server(service=service) as manager:
            async with connect(manager.endpoint) as websocket:
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
                initialize_response = parse_jsonrpc_message(await websocket.recv())
                await websocket.send(jsonrpc_request(2, "session/new", {"_meta": {"scopeId": "scope-a"}}))
                new_response = parse_jsonrpc_message(await websocket.recv())
                session_id = new_response["result"]["sessionId"]

                await websocket.send(
                    jsonrpc_request(
                        3,
                        "session/set_model",
                        {
                            "_meta": {"scopeId": "scope-a"},
                            "sessionId": session_id,
                            "model": "test-model",
                        },
                    ),
                )
                set_model_response = await _recv_response(websocket)

                await websocket.send(
                    jsonrpc_request(
                        4,
                        "session/prompt",
                        _scope_params(session_id, "server prompt"),
                    ),
                )
                prompt_response, live_updates = await _recv_response_with_updates(websocket)

                await websocket.send(
                    jsonrpc_request(5, "session/load", {"_meta": {"scopeId": "scope-a"}, "sessionId": session_id}),
                )
                replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(5)]
                parse_jsonrpc_message(await websocket.recv())

    assert initialize_response["result"]["protocolVersion"] == ACP_PROTOCOL_VERSION
    assert set_model_response["result"]["_meta"]["model"] == "test-model"
    live_payloads = [_update(message) for message in live_updates]
    live_message_payloads = [update for update in live_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _normalized_updates(live_message_payloads[:3]) == [
        {
            "sessionUpdate": "message_added",
            "messageId": "message-0",
            "message": {"role": "user", "content": "server prompt"},
        },
        {
            "sessionUpdate": "message_delta",
            "messageId": "message-1",
            "appendText": "server answer",
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "assistant", "content": "server answer"},
        },
    ]
    assert prompt_response["result"] == {"stopReason": "end_turn"}
    replay_payloads = [_update(message) for message in replay]
    assert _normalized_updates(replay_payloads) == [
        {"sessionUpdate": "history_reset"},
        {"sessionUpdate": "message_added", "messageId": "message-0", "message": {"role": "system"}},
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "user", "content": "server prompt"},
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {"role": "assistant", "content": "server answer"},
        },
        {"sessionUpdate": "history_complete"},
    ]


@pytest.mark.asyncio
async def test_remote_worker_smoke_uses_fake_openai_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with run_fake_openai_server() as fake_openai:
        monkeypatch.setenv("OPENAI_BASE_URL", fake_openai.base_url)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        runtime = WorkerRuntimeConfig(model="fake-model", tools=[])

        async with start_session_worker_server(runtime=runtime) as worker_server:
            service, store = _manager_service(tmp_path=tmp_path, endpoint=worker_server.endpoint)
            created = store.create_session(
                scope_id="scope-a",
                messages=[SystemMessage(content="system")],
                metadata={"model": "test-model"},
            )
            updates: list[SessionUpdate] = []

            async def collect_update(update: SessionUpdate) -> None:
                updates.append(update)

            result = await service.prompt(
                params=_scope_params(created.record.session_id, "container smoke"),
                on_update=collect_update,
            )
            loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert result.stop_reason == "end_turn"
    assert [getattr(message, "content", None) for message in loaded.messages] == [
        "system",
        "container smoke",
        "fake response: container smoke",
    ]
