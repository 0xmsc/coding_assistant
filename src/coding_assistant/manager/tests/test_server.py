from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from coding_assistant.core.session_updates import (
    AgentMessageChunkUpdate,
    SessionUpdate,
    ToolCallLifecycleUpdate,
    ToolCallStartedUpdate,
)
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall, ToolMessage, UserMessage
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import (
    MAX_ATTACHMENT_BYTES,
    ManagerError,
    ManagerService,
    WorkerCommit,
    WorkerPrompt,
    WorkerRunner,
)
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


async def _new_session(
    websocket: ClientConnection,
    *,
    scope_id: str,
    request_id: int = 2,
    metadata: dict[str, Any] | None = None,
) -> str:
    params = _scope_params(scope_id)
    if metadata:
        params["_meta"].update(metadata)
    await websocket.send(jsonrpc_request(request_id, "session/new", params))
    response = parse_jsonrpc_message(await websocket.recv())
    session_id = response["result"]["sessionId"]
    assert isinstance(session_id, str)
    return session_id


async def _set_model(
    websocket: ClientConnection,
    *,
    scope_id: str,
    session_id: str,
    request_id: int,
    model: str = "test-model",
) -> dict[str, Any]:
    await websocket.send(
        jsonrpc_request(
            request_id,
            "session/set_model",
            _scope_params(scope_id, sessionId=session_id, model=model),
        ),
    )
    response = parse_jsonrpc_message(await websocket.recv())
    assert "result" in response, response
    return response


async def _ignore_remote_event(event: RemoteClientEvent) -> None:
    del event


async def _ignore_disconnect(endpoint: str) -> None:
    del endpoint


async def _test_model_lister() -> list[str]:
    return ["test-model", "alternate-model"]


async def _failing_model_lister() -> list[str]:
    raise RuntimeError("model provider unavailable")


async def _ignore_session_update(update: Any) -> None:
    del update


@asynccontextmanager
async def _manager_endpoint(
    *,
    tmp_path: Path,
    worker: WorkerRunner | None = None,
    auth_secret: str | None = None,
) -> AsyncIterator[str]:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    service = ManagerService(
        store=store,
        worker_runner=worker or FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
    async with start_manager_server(
        service=service,
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
async def test_manager_lists_models_from_provider(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(jsonrpc_request(2, "model/list", {}))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["result"] == {
        "models": [{"id": "test-model"}, {"id": "alternate-model"}],
    }


@pytest.mark.asyncio
async def test_manager_model_list_is_empty_when_provider_fails_without_cache(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_failing_model_lister,
    )

    async with start_manager_server(service=service) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(jsonrpc_request(2, "model/list", {}))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["result"] == {"models": []}


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
    assert "model" not in sessions[0]["_meta"]


@pytest.mark.asyncio
async def test_manager_preserves_existing_sessions_without_model_metadata(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
    created = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])

    async with start_manager_server(service=service) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(jsonrpc_request(2, "session/list", _scope_params("scope-a")))
            list_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(
                jsonrpc_request(3, "session/load", _scope_params("scope-a", sessionId=created.record.session_id))
            )
            _ = parse_jsonrpc_message(await websocket.recv())
            _ = parse_jsonrpc_message(await websocket.recv())
            load_response = parse_jsonrpc_message(await websocket.recv())

    assert "model" not in list_response["result"]["sessions"][0]["_meta"]
    assert "model" not in load_response["result"]["_meta"]


@pytest.mark.asyncio
async def test_manager_load_replays_persisted_transcript(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(jsonrpc_request(3, "session/load", _scope_params("scope-a", sessionId=session_id)))
            reset = parse_jsonrpc_message(await websocket.recv())
            complete = parse_jsonrpc_message(await websocket.recv())
            response = parse_jsonrpc_message(await websocket.recv())

    assert _update(reset) == {"sessionUpdate": "history_reset"}
    assert _update(complete) == {"sessionUpdate": "history_complete", "version": 0}
    assert response["result"]["sessionId"] == session_id


@pytest.mark.asyncio
async def test_manager_upload_file_writes_attachment_and_replays_visible_message(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/upload_file",
                    _scope_params(
                        "scope-a",
                        sessionId=session_id,
                        name="../Meal Photo.PNG",
                        mimeType="image/png",
                        data=base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode("ascii"),
                    ),
                )
            )
            update = parse_jsonrpc_message(await websocket.recv())
            upload_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(4, "session/load", _scope_params("scope-a", sessionId=session_id)))
            reset = parse_jsonrpc_message(await websocket.recv())
            replay = parse_jsonrpc_message(await websocket.recv())
            complete = parse_jsonrpc_message(await websocket.recv())
            load_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/download_attachment",
                    _scope_params(
                        "scope-a", sessionId=session_id, attachmentId=upload_response["result"]["attachment"]["id"]
                    ),
                )
            )
            download_response = parse_jsonrpc_message(await websocket.recv())

    attachment = upload_response["result"]["attachment"]
    assert attachment["name"] == "Meal-Photo.PNG"
    assert attachment["mimeType"] == "image/png"
    assert attachment["size"] == 13
    assert attachment["path"].startswith("/attachments/att_")
    assert attachment["path"].endswith("-Meal-Photo.PNG")
    filename = attachment["path"].removeprefix("/attachments/")
    assert (tmp_path / "sessions" / session_id / "attachments" / filename).read_bytes() == b"\x89PNG\r\n\x1a\nimage"

    update_payload = _update(update)
    assert update_payload["sessionUpdate"] == "item_added"
    assert update_payload["item"]["kind"] == "attachment"
    assert _item_payload(update_payload) == attachment
    assert _update(reset) == {"sessionUpdate": "history_reset"}
    assert _update(replay)["sessionUpdate"] == "item_added"
    assert _item_payload(_update(replay)) == attachment
    assert _update(complete) == {"sessionUpdate": "history_complete", "version": 1}
    assert upload_response["result"]["session"]["_meta"]["version"] == 1
    assert load_response["result"]["_meta"]["version"] == 1
    assert download_response["result"]["attachment"] == attachment
    assert download_response["result"]["encoding"] == "base64"
    assert base64.b64decode(download_response["result"]["data"]) == b"\x89PNG\r\n\x1a\nimage"


@pytest.mark.asyncio
async def test_manager_download_attachment_checks_file_hash(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/upload_file",
                    _scope_params(
                        "scope-a",
                        sessionId=session_id,
                        name="notes.txt",
                        mimeType="text/plain",
                        data=base64.b64encode(b"original").decode("ascii"),
                    ),
                )
            )
            _ = parse_jsonrpc_message(await websocket.recv())
            upload_response = parse_jsonrpc_message(await websocket.recv())
            attachment = upload_response["result"]["attachment"]
            filename = attachment["path"].removeprefix("/attachments/")
            (tmp_path / "sessions" / session_id / "attachments" / filename).write_text("tampered", encoding="utf-8")

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/download_attachment",
                    _scope_params("scope-a", sessionId=session_id, attachmentId=attachment["id"]),
                )
            )
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == f"Attachment {attachment['id']} failed hash verification."


@pytest.mark.asyncio
async def test_manager_upload_file_uses_mime_type_name_for_unnamed_attachment(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/upload_file",
                    _scope_params(
                        "scope-a",
                        sessionId=session_id,
                        name=None,
                        mimeType="text/plain",
                        data=base64.b64encode(b"clipboard text").decode("ascii"),
                    ),
                )
            )
            update = parse_jsonrpc_message(await websocket.recv())
            upload_response = parse_jsonrpc_message(await websocket.recv())

    attachment = upload_response["result"]["attachment"]
    assert attachment["name"] == "attachment.txt"
    assert attachment["mimeType"] == "text/plain"
    assert attachment["path"].startswith("/attachments/att_")
    assert attachment["path"].endswith("-attachment.txt")
    filename = attachment["path"].removeprefix("/attachments/")
    assert (tmp_path / "sessions" / session_id / "attachments" / filename).read_text() == "clipboard text"
    update_payload = _update(update)
    assert update_payload["sessionUpdate"] == "item_added"
    assert _item_payload(update_payload) == attachment


@pytest.mark.asyncio
async def test_manager_upload_file_accepts_pdf_and_instructs_text_extraction(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
    session = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])

    result = await service.upload_file(
        params=_scope_params(
            "scope-a",
            sessionId=session.record.session_id,
            name="report.pdf",
            mimeType="",
            data=base64.b64encode(b"%PDF-1.7 report").decode("ascii"),
        ),
        on_update=_ignore_session_update,
    )

    attachment = result["attachment"]
    assert attachment["name"] == "report.pdf"
    assert attachment["mimeType"] == "application/pdf"
    assert attachment["path"].startswith("/attachments/att_")
    assert attachment["path"].endswith("-report.pdf")

    loaded = store.load_session(scope_id="scope-a", session_id=session.record.session_id)
    message = loaded.messages[-1]
    assert isinstance(message, UserMessage)
    assert isinstance(message.content, str)
    assert "pdf-text-extraction" in message.content
    assert (
        f'mkdir -p extracted && pdftotext -layout -enc UTF-8 "{attachment["path"]}" '
        f'"extracted/{attachment["id"]}-report.txt"'
    ) in message.content


@pytest.mark.asyncio
async def test_manager_upload_file_rejects_cross_scope_session(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/upload_file",
                    _scope_params(
                        "scope-b",
                        sessionId=session_id,
                        name="meal.txt",
                        mimeType="text/plain",
                        data=base64.b64encode(b"meal").decode("ascii"),
                    ),
                )
            )
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == f"Session {session_id} was not found."


@pytest.mark.asyncio
async def test_manager_upload_file_rejects_unsupported_mime_type(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/upload_file",
                    _scope_params(
                        "scope-a",
                        sessionId=session_id,
                        name="archive.zip",
                        mimeType="application/zip",
                        data=base64.b64encode(b"zip").decode("ascii"),
                    ),
                )
            )
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Unsupported attachment MIME type: application/zip."


@pytest.mark.asyncio
async def test_manager_upload_file_rejects_too_large_attachment(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
    session = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])

    with pytest.raises(ManagerError, match="too large"):
        await service.upload_file(
            params=_scope_params(
                "scope-a",
                sessionId=session.record.session_id,
                name="large.txt",
                mimeType="text/plain",
                data=base64.b64encode(b"x" * (MAX_ATTACHMENT_BYTES + 1)).decode("ascii"),
            ),
            on_update=_ignore_session_update,
        )


@pytest.mark.asyncio
async def test_manager_load_replays_persisted_tool_calls(tmp_path: Path) -> None:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "sessions"),
    )
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
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

    async with start_manager_server(service=service) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(
                jsonrpc_request(2, "session/load", _scope_params("scope-a", sessionId=created.record.session_id))
            )
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(5)]
            response = parse_jsonrpc_message(await websocket.recv())

    replay_updates = [message["params"]["update"] for message in replay]
    assert replay_updates[0] == {"sessionUpdate": "history_reset"}
    assert replay_updates[1]["sessionUpdate"] == "item_added"
    assert _item_payload(replay_updates[1]) == {"role": "user", "content": "Read smoke.txt"}
    assert replay_updates[2]["sessionUpdate"] == "item_added"
    assert _item_payload(replay_updates[2]) == {
        "toolCallId": "call-1",
        "title": "shell_execute",
        "kind": "other",
        "status": "completed",
        "rawInput": {"command": "cat smoke.txt"},
        "rawOutput": "smoke output",
        "content": "smoke output",
    }
    assert replay_updates[3]["sessionUpdate"] == "item_added"
    assert _item_payload(replay_updates[3]) == {"role": "assistant", "content": "Done"}
    assert replay_updates[4] == {"sessionUpdate": "history_complete", "version": 1}
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
async def test_manager_sets_session_model_and_uses_it_for_prompt(tmp_path: Path) -> None:
    worker = FakeWorkerRunner(response_text="done")
    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/set_model",
                    _scope_params("scope-a", sessionId=session_id, model="alternate-model"),
                ),
            )
            set_model_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("Do it")]),
                ),
            )
            live_updates = [parse_jsonrpc_message(await websocket.recv()) for _ in range(3)]
            prompt_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(6, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(4)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    assert set_model_response["result"]["_meta"]["model"] == "alternate-model"
    live_payloads = [_update(message) for message in live_updates]
    assert _item_payload(live_payloads[0]) == {"role": "user", "content": "Do it"}
    assert _item_payload(live_payloads[1]) == {"role": "assistant", "content": ""}
    assert live_payloads[2]["sessionUpdate"] == "item_delta"
    assert live_payloads[2]["appendText"] == "done"
    assert prompt_response["result"] == {"stopReason": "end_turn"}
    replay_payloads = [_update(message) for message in replay]
    assert replay_payloads[0] == {"sessionUpdate": "history_reset"}
    assert _item_payload(replay_payloads[1]) == {"role": "user", "content": "Do it"}
    assert _item_payload(replay_payloads[2]) == {"role": "assistant", "content": "done"}
    assert replay_payloads[3] == {"sessionUpdate": "history_complete", "version": 1}
    assert load_response["result"]["_meta"]["model"] == "alternate-model"
    assert worker.prompts is not None
    assert worker.prompts[0].model == "alternate-model"


@pytest.mark.asyncio
async def test_manager_stores_session_worker_env_privately_and_injects_session_skills(tmp_path: Path) -> None:
    worker = FakeWorkerRunner(response_text="done")
    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            new_session_params = _scope_params("scope-a")
            new_session_params["_meta"]["workerEnv"] = {
                "APPS_API_BASE_URL": "http://apps-api",
                "APPS_API_TOKEN": "secret-token",
            }
            new_session_params["_meta"]["skills"] = [
                {
                    "name": "apps-api",
                    "description": "Use apps REST APIs.",
                    "files": {
                        "SKILL.md": "---\nname: apps-api\ndescription: Use apps REST APIs.\n---\n",
                        "references/calories.md": "calories",
                    },
                },
            ]
            await websocket.send(jsonrpc_request(2, "session/new", new_session_params))
            new_session_response = parse_jsonrpc_message(await websocket.recv())
            session_id = new_session_response["result"]["sessionId"]
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)
            params = _scope_params(
                "scope-a",
                sessionId=session_id,
                prompt=[text_block("Do it")],
            )

            await websocket.send(jsonrpc_request(4, "session/prompt", params))
            live_updates = [parse_jsonrpc_message(await websocket.recv()) for _ in range(3)]
            prompt_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(4)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    live_payloads = [_update(message) for message in live_updates]
    assert _item_payload(live_payloads[0]) == {"role": "user", "content": "Do it"}
    assert _item_payload(live_payloads[1]) == {"role": "assistant", "content": ""}
    assert live_payloads[2]["sessionUpdate"] == "item_delta"
    assert live_payloads[2]["appendText"] == "done"
    assert prompt_response["result"] == {"stopReason": "end_turn"}
    replay_payloads = [_update(message) for message in replay]
    assert replay_payloads[0] == {"sessionUpdate": "history_reset"}
    assert _item_payload(replay_payloads[1]) == {"role": "user", "content": "Do it"}
    assert _item_payload(replay_payloads[2]) == {"role": "assistant", "content": "done"}
    assert replay_payloads[3] == {"sessionUpdate": "history_complete", "version": 1}
    assert "capabilities" not in load_response["result"]["_meta"]
    assert "skills" not in load_response["result"]["_meta"]
    assert "workerEnv" not in load_response["result"]["_meta"]
    skill_root = tmp_path / "sessions" / session_id / "workspace" / ".agents" / "skills" / "apps-api"
    assert (skill_root / "SKILL.md").read_text(encoding="utf-8") == (
        "---\nname: apps-api\ndescription: Use apps REST APIs.\n---\n"
    )
    assert (skill_root / "references" / "calories.md").read_text(encoding="utf-8") == "calories"
    assert worker.prompts is not None
    assert worker.prompts[0].worker_env == {
        "APPS_API_BASE_URL": "http://apps-api",
        "APPS_API_TOKEN": "secret-token",
    }
    system_message = worker.prompts[0].history[0]
    assert isinstance(system_message, SystemMessage)
    assert isinstance(system_message.content, str)
    assert "apps-api" in system_message.content
    assert "Use apps REST APIs." in system_message.content


@pytest.mark.asyncio
async def test_manager_ignores_unknown_capabilities_metadata(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            params = _scope_params("scope-a")
            params["_meta"]["capabilities"] = {"workerEnv": {"APPS_API_TOKEN": "secret"}}

            await websocket.send(jsonrpc_request(2, "session/list", params))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["result"]["sessions"] == []


@pytest.mark.asyncio
async def test_manager_validates_session_worker_env(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            params = _scope_params("scope-a")
            params["_meta"]["workerEnv"] = {"not-valid": "value"}

            await websocket.send(jsonrpc_request(2, "session/new", params))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Invalid worker environment variable name: 'not-valid'."


@pytest.mark.asyncio
async def test_manager_ignores_prompt_worker_env(tmp_path: Path) -> None:
    worker = FakeWorkerRunner(response_text="done")
    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)
            params = _scope_params(
                "scope-a",
                sessionId=session_id,
                prompt=[text_block("Do it")],
            )
            params["_meta"]["capabilities"] = {
                "workerEnv": {"APPS_API_TOKEN": "secret-token"},
            }

            await websocket.send(jsonrpc_request(4, "session/prompt", params))
            live_updates = [parse_jsonrpc_message(await websocket.recv()) for _ in range(3)]
            response = parse_jsonrpc_message(await websocket.recv())

    live_payloads = [_update(message) for message in live_updates]
    assert _item_payload(live_payloads[0]) == {"role": "user", "content": "Do it"}
    assert live_payloads[2]["appendText"] == "done"
    assert response["result"] == {"stopReason": "end_turn"}
    assert worker.prompts is not None
    assert worker.prompts[0].worker_env == {}


@pytest.mark.asyncio
async def test_manager_validates_injected_skill_paths(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            params = _scope_params("scope-a")
            params["_meta"]["skills"] = [
                {
                    "name": "apps-api",
                    "description": "Use apps REST APIs.",
                    "files": {
                        "SKILL.md": "skill",
                        "../escape.md": "bad",
                    },
                },
            ]

            await websocket.send(jsonrpc_request(2, "session/new", params))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Injected skill apps-api has an invalid file path: '../escape.md'."


@pytest.mark.asyncio
async def test_manager_rejects_unavailable_session_model(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(
                jsonrpc_request(
                    3,
                    "session/set_model",
                    _scope_params("scope-a", sessionId=session_id, model="missing-model"),
                ),
            )
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Model missing-model is not available."


@pytest.mark.asyncio
async def test_manager_rejects_prompt_without_session_model(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
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
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Session has no model selected."


@pytest.mark.asyncio
async def test_manager_rejects_model_change_during_active_prompt(tmp_path: Path) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    worker = FakeWorkerRunner(started=started, release=release)

    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("first")]),
                ),
            )
            first_updates = [parse_jsonrpc_message(await websocket.recv()) for _ in range(3)]
            await asyncio.wait_for(started.wait(), timeout=1)

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/set_model",
                    _scope_params("scope-a", sessionId=session_id, model="alternate-model"),
                ),
            )
            busy_response = parse_jsonrpc_message(await websocket.recv())
            release.set()
            first_response = parse_jsonrpc_message(await websocket.recv())

    assert _item_payload(_update(first_updates[0])) == {"role": "user", "content": "first"}
    assert _update(first_updates[2])["appendText"] == "fake response"
    assert busy_response["error"]["message"] == "Cannot change model while session has an active prompt."
    assert first_response["result"] == {"stopReason": "end_turn"}


@pytest.mark.asyncio
async def test_manager_prompt_streams_update_and_commits_messages(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path, worker=FakeWorkerRunner(response_text="done")) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("Do it")]),
                ),
            )
            live_updates = [parse_jsonrpc_message(await websocket.recv()) for _ in range(3)]
            response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay_messages = [parse_jsonrpc_message(await websocket.recv()) for _ in range(4)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    live_payloads = [_update(message) for message in live_updates]
    assert _item_payload(live_payloads[0]) == {"role": "user", "content": "Do it"}
    assert _item_payload(live_payloads[1]) == {"role": "assistant", "content": ""}
    assert live_payloads[2]["sessionUpdate"] == "item_delta"
    assert live_payloads[2]["appendText"] == "done"
    assert response["result"] == {"stopReason": "end_turn"}
    replay_payloads = [_update(message) for message in replay_messages]
    assert replay_payloads[0] == {"sessionUpdate": "history_reset"}
    assert _item_payload(replay_payloads[1]) == {"role": "user", "content": "Do it"}
    assert _item_payload(replay_payloads[2]) == {"role": "assistant", "content": "done"}
    assert replay_payloads[3] == {"sessionUpdate": "history_complete", "version": 1}
    assert load_response["result"]["_meta"]["version"] == 1


@pytest.mark.asyncio
async def test_manager_translates_live_tool_updates_to_session_items(tmp_path: Path) -> None:
    class ToolStreamingWorker:
        async def run_prompt(
            self,
            *,
            prompt: WorkerPrompt,
            on_update: Callable[[SessionUpdate], Awaitable[None]],
        ) -> WorkerCommit:
            del prompt
            await on_update(
                ToolCallStartedUpdate(
                    source="worker",
                    tool_call_id="call-1",
                    title="shell_execute",
                    tool_kind="other",
                    status="pending",
                    raw_input={"command": "cat smoke.txt"},
                    message=None,
                )
            )
            await on_update(
                ToolCallLifecycleUpdate(
                    source="worker",
                    tool_call_id="call-1",
                    title="shell_execute",
                    tool_kind="other",
                    status="completed",
                    raw_input={"command": "cat smoke.txt"},
                    raw_output="smoke output",
                    content="smoke output",
                )
            )
            await on_update(AgentMessageChunkUpdate(content="done"))
            return WorkerCommit(
                messages=[
                    UserMessage(content="Use tool"),
                    AssistantMessage(content="done"),
                ],
                stop_reason="end_turn",
            )

        async def cancel(self, *, session_id: str) -> None:
            del session_id

    async with _manager_endpoint(tmp_path=tmp_path, worker=ToolStreamingWorker()) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("Use tool")]),
                ),
            )
            live_messages = [parse_jsonrpc_message(await websocket.recv()) for _ in range(5)]
            response = parse_jsonrpc_message(await websocket.recv())

    live_payloads = [_update(message) for message in live_messages]
    assert _item_payload(live_payloads[0]) == {"role": "user", "content": "Use tool"}
    assert live_payloads[1]["sessionUpdate"] == "item_added"
    assert live_payloads[1]["item"]["kind"] == "tool_call"
    tool_item_id = live_payloads[1]["item"]["id"]
    assert _item_payload(live_payloads[1]) == {
        "toolCallId": "call-1",
        "title": "shell_execute",
        "kind": "other",
        "status": "pending",
        "rawInput": {"command": "cat smoke.txt"},
    }
    assert live_payloads[2] == {
        "sessionUpdate": "item_updated",
        "itemId": tool_item_id,
        "patch": {
            "payload": {
                "status": "completed",
                "title": "shell_execute",
                "kind": "other",
                "rawInput": {"command": "cat smoke.txt"},
                "rawOutput": "smoke output",
                "content": "smoke output",
            }
        },
    }
    assert _item_payload(live_payloads[3]) == {"role": "assistant", "content": ""}
    assert live_payloads[4]["sessionUpdate"] == "item_delta"
    assert live_payloads[4]["itemId"] == live_payloads[3]["item"]["id"]
    assert live_payloads[4]["appendText"] == "done"
    assert response["result"] == {"stopReason": "end_turn"}


@pytest.mark.asyncio
async def test_manager_rejects_cross_scope_load_rename_model_prompt_and_cancel(tmp_path: Path) -> None:
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
                    "session/set_model",
                    _scope_params("scope-b", sessionId=session_id, model="alternate-model"),
                ),
            )
            set_model_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(
                jsonrpc_request(
                    6,
                    "session/prompt",
                    _scope_params("scope-b", sessionId=session_id, prompt=[text_block("bad")]),
                ),
            )
            prompt_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(7, "session/cancel", _scope_params("scope-b", sessionId=session_id)))
            cancel_response = parse_jsonrpc_message(await websocket.recv())

    assert load_response["error"]["code"] == -32602
    assert rename_response["error"]["code"] == -32602
    assert set_model_response["error"]["code"] == -32602
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
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("first")]),
                ),
            )
            first_updates = [parse_jsonrpc_message(await websocket.recv()) for _ in range(3)]
            await asyncio.wait_for(started.wait(), timeout=1)

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("second")]),
                ),
            )
            busy_response = parse_jsonrpc_message(await websocket.recv())
            release.set()
            first_response = parse_jsonrpc_message(await websocket.recv())

    assert _item_payload(_update(first_updates[0])) == {"role": "user", "content": "first"}
    assert _update(first_updates[2])["appendText"] == "fake response"
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
            await _set_model(first, scope_id="scope-a", session_id=first_session_id, request_id=3)
            await _set_model(second, scope_id="scope-a", session_id=second_session_id, request_id=3)

            await first.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=first_session_id, prompt=[text_block("first")]),
                ),
            )
            await second.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=second_session_id, prompt=[text_block("second")]),
                ),
            )
            first_updates = [parse_jsonrpc_message(await first.recv()) for _ in range(3)]
            second_updates = [parse_jsonrpc_message(await second.recv()) for _ in range(3)]
            release.set()
            first_response = parse_jsonrpc_message(await first.recv())
            second_response = parse_jsonrpc_message(await second.recv())

    assert first_updates[0]["params"]["sessionId"] == first_session_id
    assert second_updates[0]["params"]["sessionId"] == second_session_id
    assert first_response["result"] == {"stopReason": "end_turn"}
    assert second_response["result"] == {"stopReason": "end_turn"}
