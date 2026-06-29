from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect

from coding_assistant.core.session_updates import SessionItemAddedUpdate, SessionItemDeltaUpdate, SessionUpdate
from coding_assistant.llm.types import (
    AssistantMessage,
    Completion,
    CompletionEvent,
    ContentDeltaEvent,
    SystemMessage,
    Usage,
)
from coding_assistant.manager.remote_worker import RemoteWorkerRunner
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import ManagerService
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.workspace import WorkspacePaths
from coding_assistant.remote.acp import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block
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


def _item_payload(update: dict[str, Any]) -> dict[str, Any]:
    item = update["item"]
    assert isinstance(item, dict)
    payload = item["payload"]
    assert isinstance(payload, dict)
    return payload


async def _test_model_lister() -> list[str]:
    return ["test-model"]


def _manager_service(*, tmp_path: Path, endpoint: str) -> tuple[ManagerService, SessionStore]:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    return ManagerService(
        store=store,
        worker_runner=RemoteWorkerRunner(endpoint=endpoint),
        model_lister=_test_model_lister,
    ), store


async def _ignore_update(update: SessionUpdate) -> None:
    del update


@pytest.mark.asyncio
async def test_remote_worker_prompt_streams_update_and_commits_to_sqlite(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer([AssistantMessage(content="hello")]),
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
    assert len(updates) == 3
    assert isinstance(updates[0], SessionItemAddedUpdate)
    assert updates[0].item.payload == {"role": "user", "content": "Do it"}
    assert isinstance(updates[1], SessionItemAddedUpdate)
    assert updates[1].item.payload == {"role": "assistant", "content": ""}
    assert isinstance(updates[2], SessionItemDeltaUpdate)
    assert updates[2].append_text == "hello"
    assert loaded.record.version == 1
    assert [message.role for message in loaded.messages] == ["system", "user", "assistant"]
    assert [getattr(message, "content", None) for message in loaded.messages] == ["system", "Do it", "hello"]


@pytest.mark.asyncio
async def test_remote_worker_two_sequential_prompts_advance_history_once(tmp_path: Path) -> None:
    runtime = WorkerRuntimeConfig(
        model="test-model",
        tools=[],
        completion_streamer=ScriptedStreamer(
            [
                AssistantMessage(content="first answer"),
                AssistantMessage(content="second answer"),
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

        first = await service.prompt(
            params=_scope_params(created.record.session_id, "first"),
            on_update=lambda update: _ignore_update(update),
        )
        second = await service.prompt(
            params=_scope_params(created.record.session_id, "second"),
            on_update=lambda update: _ignore_update(update),
        )
        loaded = store.load_session(scope_id="scope-a", session_id=created.record.session_id)

    assert first.stop_reason == "end_turn"
    assert second.stop_reason == "end_turn"
    assert loaded.record.version == 2
    assert [message.role for message in loaded.messages] == ["system", "user", "assistant", "user", "assistant"]
    assert [getattr(message, "content", None) for message in loaded.messages] == [
        "system",
        "first",
        "first answer",
        "second",
        "second answer",
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
    assert loaded.record.version == 1
    assert [message.role for message in loaded.messages] == ["system", "user"]


@pytest.mark.asyncio
async def test_manager_server_uses_remote_worker_and_replays_committed_history(tmp_path: Path) -> None:
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
                set_model_response = parse_jsonrpc_message(await websocket.recv())

                await websocket.send(
                    jsonrpc_request(
                        4,
                        "session/prompt",
                        _scope_params(session_id, "server prompt"),
                    ),
                )
                live_updates = [parse_jsonrpc_message(await websocket.recv()) for _ in range(3)]
                prompt_response = parse_jsonrpc_message(await websocket.recv())

                await websocket.send(
                    jsonrpc_request(5, "session/load", {"_meta": {"scopeId": "scope-a"}, "sessionId": session_id}),
                )
                replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(4)]
                load_response = parse_jsonrpc_message(await websocket.recv())

    assert initialize_response["result"]["protocolVersion"] == ACP_PROTOCOL_VERSION
    assert set_model_response["result"]["_meta"]["model"] == "test-model"
    live_payloads = [_update(message) for message in live_updates]
    assert _item_payload(live_payloads[0]) == {"role": "user", "content": "server prompt"}
    assert _item_payload(live_payloads[1]) == {"role": "assistant", "content": ""}
    assert live_payloads[2]["appendText"] == "server answer"
    assert prompt_response["result"] == {"stopReason": "end_turn"}
    replay_payloads = [_update(message) for message in replay]
    assert replay_payloads[0] == {"sessionUpdate": "history_reset"}
    assert _item_payload(replay_payloads[1]) == {"role": "user", "content": "server prompt"}
    assert _item_payload(replay_payloads[2]) == {"role": "assistant", "content": "server answer"}
    assert replay_payloads[3] == {"sessionUpdate": "history_complete", "version": 1}
    assert load_response["result"]["_meta"]["version"] == 1


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
    assert loaded.record.version == 1
    assert [getattr(message, "content", None) for message in loaded.messages] == [
        "system",
        "container smoke",
        "fake response: container smoke",
    ]
