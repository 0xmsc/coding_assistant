from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall, ToolMessage, UserMessage
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import ManagerService
from coding_assistant.manager.store import SessionStore
from coding_assistant.manager.tests.fakes import FakeWorkerRunner
from coding_assistant.manager.workspace import WorkspacePaths
from coding_assistant.remote.acp import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block
from coding_assistant.remote.client import RemoteClientEvent, RemoteSessionClient


def _scope_params(scope_id: str, **params: Any) -> dict[str, Any]:
    return {
        "_meta": {
            "scopeId": scope_id,
        },
        **params,
    }


async def _initialize(websocket: ClientConnection) -> dict[str, Any]:
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
    return parse_jsonrpc_message(await websocket.recv())


async def _new_session(websocket: ClientConnection, *, scope_id: str, request_id: int = 2) -> str:
    await websocket.send(jsonrpc_request(request_id, "session/new", _scope_params(scope_id)))
    response = parse_jsonrpc_message(await websocket.recv())
    session_id = response["result"]["sessionId"]
    assert isinstance(session_id, str)
    return session_id


async def _ignore_remote_event(event: RemoteClientEvent) -> None:
    del event


async def _ignore_disconnect(endpoint: str) -> None:
    del endpoint


@asynccontextmanager
async def _manager_endpoint(
    *,
    tmp_path: Path,
    worker: FakeWorkerRunner | None = None,
    auth_secret: str | None = None,
) -> AsyncIterator[str]:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    service = ManagerService(store=store, worker_runner=worker or FakeWorkerRunner())
    async with start_manager_server(
        service=service,
        initial_messages=[SystemMessage(content="system")],
        auth_secret=auth_secret,
    ) as server:
        yield server.endpoint


@pytest.mark.asyncio
async def test_manager_auth_rejects_missing_bearer_token(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path, auth_secret="secret-token") as endpoint:
        with pytest.raises(InvalidStatus) as error:
            async with connect(endpoint):
                pass

    assert error.value.response.status_code == 401


@pytest.mark.asyncio
async def test_manager_auth_rejects_wrong_bearer_token(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path, auth_secret="secret-token") as endpoint:
        with pytest.raises(InvalidStatus) as error:
            async with connect(endpoint, additional_headers={"Authorization": "Bearer wrong"}):
                pass

    assert error.value.response.status_code == 401


@pytest.mark.asyncio
async def test_manager_auth_accepts_correct_bearer_token(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path, auth_secret="secret-token") as endpoint:
        async with connect(endpoint, additional_headers={"Authorization": "Bearer secret-token"}) as websocket:
            response = await _initialize(websocket)

    assert response["result"]["agentCapabilities"]["loadSession"] is True


@pytest.mark.asyncio
async def test_remote_session_client_sends_manager_auth_token(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path, auth_secret="secret-token") as endpoint:
        client = await RemoteSessionClient.connect(
            endpoint=endpoint,
            auth_token="secret-token",
            on_event=_ignore_remote_event,
            on_disconnect=_ignore_disconnect,
        )
        try:
            await client.initialize()
            session_id = await client.new_session(_scope_params("scope-a"))
        finally:
            await client.close()

    assert session_id.startswith("sess_")


@pytest.mark.asyncio
async def test_manager_requires_initialize_before_session_methods(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await websocket.send(jsonrpc_request(1, "session/list", _scope_params("scope-a")))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32600
    assert response["error"]["message"] == "initialize must be called first."


@pytest.mark.asyncio
async def test_manager_creates_and_lists_sessions_by_scope(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            assert (await _initialize(websocket))["result"]["agentCapabilities"]["loadSession"] is True
            session_id = await _new_session(websocket, scope_id="scope-a")
            await _new_session(websocket, scope_id="scope-b", request_id=3)

            await websocket.send(jsonrpc_request(4, "session/list", _scope_params("scope-a")))
            response = parse_jsonrpc_message(await websocket.recv())

    sessions = response["result"]["sessions"]
    assert [session["sessionId"] for session in sessions] == [session_id]
    assert sessions[0]["_meta"]["version"] == 0


@pytest.mark.asyncio
async def test_manager_load_replays_persisted_transcript(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(jsonrpc_request(3, "session/load", _scope_params("scope-a", sessionId=session_id)))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["result"]["sessionId"] == session_id


@pytest.mark.asyncio
async def test_manager_load_replays_persisted_tool_calls(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    service = ManagerService(store=store, worker_runner=FakeWorkerRunner())
    created = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="shell_execute", arguments='{"command": "cat smoke.txt"}'),
    )
    store.commit_messages(
        scope_id="scope-a",
        session_id=created.record.session_id,
        base_version=0,
        messages=[
            UserMessage(content="Read smoke.txt"),
            AssistantMessage(tool_calls=[tool_call]),
            ToolMessage(tool_call_id="call-1", name="shell_execute", content="smoke output"),
            AssistantMessage(content="Done"),
        ],
    )

    async with start_manager_server(service=service, initial_messages=[SystemMessage(content="system")]) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(
                jsonrpc_request(2, "session/load", _scope_params("scope-a", sessionId=created.record.session_id))
            )
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(4)]
            response = parse_jsonrpc_message(await websocket.recv())

    replay_updates = [message["params"]["update"] for message in replay]
    assert replay_updates == [
        {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "Read smoke.txt"}},
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "call-1",
            "title": "shell_execute",
            "kind": "other",
            "status": "pending",
            "rawInput": {"command": "cat smoke.txt"},
        },
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "call-1",
            "status": "completed",
            "title": "shell_execute",
            "kind": "other",
            "rawOutput": "smoke output",
            "content": [{"type": "content", "content": {"type": "text", "text": "smoke output"}}],
        },
        {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Done"}},
    ]
    assert response["result"]["sessionId"] == created.record.session_id


@pytest.mark.asyncio
async def test_manager_renames_session(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(3, "session/rename", _scope_params("scope-a", sessionId=session_id, title="Renamed"))
            )
            rename_response = parse_jsonrpc_message(await websocket.recv())
            await websocket.send(jsonrpc_request(4, "session/list", _scope_params("scope-a")))
            list_response = parse_jsonrpc_message(await websocket.recv())

    assert rename_response["result"]["sessionId"] == session_id
    assert rename_response["result"]["title"] == "Renamed"
    assert list_response["result"]["sessions"][0]["title"] == "Renamed"


@pytest.mark.asyncio
async def test_manager_prompt_streams_update_and_commits_messages(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path, worker=FakeWorkerRunner(response_text="done")) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("Do it")]),
                ),
            )
            update = parse_jsonrpc_message(await websocket.recv())
            response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(4, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay_messages = [parse_jsonrpc_message(await websocket.recv()) for _ in range(2)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    assert update["params"]["update"]["content"]["text"] == "done"
    assert response["result"] == {"stopReason": "end_turn"}
    replay_texts = [message["params"]["update"]["content"]["text"] for message in replay_messages]
    assert replay_texts == ["Do it", "done"]
    assert load_response["result"]["_meta"]["version"] == 1


@pytest.mark.asyncio
async def test_manager_rejects_cross_scope_load_rename_prompt_and_cancel(tmp_path: Path) -> None:
    worker = FakeWorkerRunner()
    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(jsonrpc_request(3, "session/load", _scope_params("scope-b", sessionId=session_id)))
            load_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(
                jsonrpc_request(4, "session/rename", _scope_params("scope-b", sessionId=session_id, title="Bad"))
            )
            rename_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/prompt",
                    _scope_params("scope-b", sessionId=session_id, prompt=[text_block("bad")]),
                ),
            )
            prompt_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(6, "session/cancel", _scope_params("scope-b", sessionId=session_id)))
            cancel_response = parse_jsonrpc_message(await websocket.recv())

    assert load_response["error"]["code"] == -32602
    assert rename_response["error"]["code"] == -32602
    assert prompt_response["error"]["code"] == -32602
    assert cancel_response["error"]["code"] == -32602
    assert worker.cancelled_session_ids is None


@pytest.mark.asyncio
async def test_manager_rejects_missing_scope_id(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(jsonrpc_request(2, "session/list", {}))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Request params must include _meta.scopeId."


@pytest.mark.asyncio
async def test_manager_rejects_second_active_prompt_for_same_session(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    worker = FakeWorkerRunner(started=started, release=release)

    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("first")]),
                ),
            )
            first_update = parse_jsonrpc_message(await websocket.recv())
            await asyncio.wait_for(started.wait(), timeout=1)

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("second")]),
                ),
            )
            busy_response = parse_jsonrpc_message(await websocket.recv())
            release.set()
            first_response = parse_jsonrpc_message(await websocket.recv())

    assert first_update["params"]["update"]["content"]["text"] == "fake response"
    assert busy_response["error"]["message"] == "Session already has an active prompt."
    assert first_response["result"] == {"stopReason": "end_turn"}


@pytest.mark.asyncio
async def test_manager_allows_concurrent_prompts_for_different_sessions(tmp_path: Path) -> None:
    release = asyncio.Event()
    worker = FakeWorkerRunner(release=release)

    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as first, connect(endpoint) as second:
            await _initialize(first)
            await _initialize(second)
            first_session_id = await _new_session(first, scope_id="scope-a")
            second_session_id = await _new_session(second, scope_id="scope-a")

            await first.send(
                jsonrpc_request(
                    3,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=first_session_id, prompt=[text_block("first")]),
                ),
            )
            await second.send(
                jsonrpc_request(
                    3,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=second_session_id, prompt=[text_block("second")]),
                ),
            )
            first_update = parse_jsonrpc_message(await first.recv())
            second_update = parse_jsonrpc_message(await second.recv())
            release.set()
            first_response = parse_jsonrpc_message(await first.recv())
            second_response = parse_jsonrpc_message(await second.recv())

    assert first_update["params"]["sessionId"] == first_session_id
    assert second_update["params"]["sessionId"] == second_session_id
    assert first_response["result"] == {"stopReason": "end_turn"}
    assert second_response["result"] == {"stopReason": "end_turn"}
