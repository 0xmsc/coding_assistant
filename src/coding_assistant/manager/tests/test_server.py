from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from coding_assistant.core.session_updates import (
    MessageAddedUpdate,
    MessageDeltaUpdate,
    SessionUpdate,
)
from coding_assistant.llm.openai import ProviderModel
from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall, ToolMessage, UserMessage
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import (
    MAX_ATTACHMENT_BYTES,
    ManagerError,
    ManagerService,
    WorkerPrompt,
    WorkerRunResult,
    WorkerRunner,
)
from coding_assistant.manager.tests.fakes import FakeWorkerRunner
from coding_assistant.manager.tests.store_helpers import create_session_store
from coding_assistant.remote.jsonrpc import (
    ACP_PROTOCOL_VERSION,
    jsonrpc_notification,
    jsonrpc_request,
    parse_jsonrpc_message,
    text_block,
)
from coding_assistant.remote.client import RemoteSessionClient


IMAGE_BLOCK: dict[str, Any] = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


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


def _message_payload(update: dict[str, Any]) -> dict[str, Any]:
    message = update["message"]
    assert isinstance(message, dict)
    payload = dict(message)
    payload.pop("id", None)
    payload.pop("createdAt", None)
    if payload.get("role") == "system":
        return {"role": "system"}
    return payload


def _attachment_payload(update: dict[str, Any]) -> dict[str, Any]:
    attachment = update["attachment"]
    assert isinstance(attachment, dict)
    payload = dict(attachment)
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
        if update_type == "attachment_added":
            result.append({"sessionUpdate": "attachment_added", "attachment": _attachment_payload(update)})
            continue
        result.append(dict(update))
    return result


async def _recv_response(websocket: ClientConnection) -> dict[str, Any]:
    while True:
        message = parse_jsonrpc_message(await websocket.recv())
        if message.get("method") != "session/update":
            return message


async def _recv_response_with_updates(websocket: ClientConnection) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    updates: list[dict[str, Any]] = []
    while True:
        message = parse_jsonrpc_message(await websocket.recv())
        if message.get("method") == "session/update":
            updates.append(message)
            continue
        return message, updates


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
    response = await _recv_response(websocket)
    assert "result" in response, response
    return response


async def _test_model_lister() -> list[ProviderModel]:
    return [
        ProviderModel(id="test-model"),
        ProviderModel(id="alternate-model", reasoning_efforts=("low", "high")),
    ]


async def _failing_model_lister() -> list[ProviderModel]:
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
    store = create_session_store(tmp_path)
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
async def test_manager_rejects_incompatible_protocol_version_without_initializing(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await websocket.send(jsonrpc_request(1, "initialize", {"protocolVersion": 0}))
            initialize_error = parse_jsonrpc_message(await websocket.recv())
            await websocket.send(jsonrpc_request(2, "session/list", _scope_params("scope-a")))
            session_error = parse_jsonrpc_message(await websocket.recv())

    assert initialize_error["error"]["code"] == -32602
    assert "No compatible protocol version" in initialize_error["error"]["message"]
    assert session_error["error"]["message"] == "initialize must be called first."


@pytest.mark.asyncio
async def test_manager_ignores_request_only_notification_without_side_effects(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(jsonrpc_notification("session/new", _scope_params("scope-a")))
            await websocket.send(jsonrpc_request(2, "session/list", _scope_params("scope-a")))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["id"] == 2
    assert response["result"]["sessions"] == []


@pytest.mark.asyncio
async def test_manager_lists_models_from_provider(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(jsonrpc_request(2, "model/list", {}))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["result"] == {
        "models": [
            {"id": "test-model"},
            {
                "id": "alternate-model",
                "reasoningEfforts": ["low", "high"],
            },
        ],
    }


@pytest.mark.asyncio
async def test_manager_stores_and_sends_model_spec(tmp_path: Path) -> None:
    worker = FakeWorkerRunner()
    service = ManagerService(
        store=create_session_store(tmp_path),
        worker_runner=worker,
    )
    session_id = service.new_session(params=_scope_params("scope-a"))["sessionId"]

    session = await service.set_session_model(
        params=_scope_params("scope-a", sessionId=session_id, model="reasoning-model (provider-level)"),
        on_update=_ignore_session_update,
    )
    assert session["_meta"] == {"model": "reasoning-model (provider-level)"}

    await service.prompt(
        params=_scope_params("scope-a", sessionId=session_id, prompt=[text_block("Do it")]),
        on_update=_ignore_session_update,
    )
    assert worker.prompts is not None
    assert worker.prompts[0].model == "reasoning-model (provider-level)"


@pytest.mark.asyncio
async def test_manager_model_list_is_empty_when_provider_fails_without_cache(tmp_path: Path) -> None:
    store = create_session_store(tmp_path)
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
    assert sessions[0]["_meta"] == {}
    assert "model" not in sessions[0]["_meta"]


@pytest.mark.asyncio
async def test_manager_does_not_publish_updates_across_scopes(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as owner, connect(endpoint) as other_scope:
            await _initialize(owner)
            await _initialize(other_scope)
            session_id = await _new_session(owner, scope_id="scope-a")

            await other_scope.send(
                jsonrpc_request(2, "session/load", _scope_params("scope-b", sessionId=session_id)),
            )
            rejected = parse_jsonrpc_message(await other_scope.recv())
            assert rejected["error"]["code"] == -32602

            await owner.send(
                jsonrpc_request(
                    3,
                    "session/rename",
                    _scope_params("scope-a", sessionId=session_id, title="Private title"),
                ),
            )
            response, updates = await _recv_response_with_updates(owner)

            assert response["result"]["title"] == "Private title"
            assert any(_update(message).get("sessionUpdate") == "session_updated" for message in updates)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(other_scope.recv(), timeout=0.05)


@pytest.mark.asyncio
async def test_manager_preserves_existing_sessions_without_model_metadata(tmp_path: Path) -> None:
    store = create_session_store(tmp_path)
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
            system_message = parse_jsonrpc_message(await websocket.recv())
            complete = parse_jsonrpc_message(await websocket.recv())
            response = parse_jsonrpc_message(await websocket.recv())

    assert _update(reset) == {"sessionUpdate": "history_reset"}
    assert _message_payload(_update(system_message))["role"] == "system"
    assert _update(complete) == {"sessionUpdate": "history_complete"}
    assert response["result"]["sessionId"] == session_id


@pytest.mark.asyncio
async def test_manager_upload_file_writes_attachment_and_replays_upload_message(tmp_path: Path) -> None:
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
            upload_response, upload_updates = await _recv_response_with_updates(websocket)
            upload_message = upload_updates[0]
            update = upload_updates[1]
            session_update = upload_updates[2]

            await websocket.send(jsonrpc_request(4, "session/load", _scope_params("scope-a", sessionId=session_id)))
            reset = parse_jsonrpc_message(await websocket.recv())
            system_replay = parse_jsonrpc_message(await websocket.recv())
            message_replay = parse_jsonrpc_message(await websocket.recv())
            attachment_replay = parse_jsonrpc_message(await websocket.recv())
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

    upload_message_payload = _message_payload(_update(upload_message))
    assert upload_message_payload["role"] == "user"
    assert "Attached file `Meal-Photo.PNG`" in upload_message_payload["content"]
    assert _attachment_payload(_update(update)) == attachment
    assert _update(session_update)["sessionUpdate"] == "session_updated"
    assert _update(reset) == {"sessionUpdate": "history_reset"}
    assert _message_payload(_update(system_replay))["role"] == "system"
    assert _message_payload(_update(message_replay)) == upload_message_payload
    assert _attachment_payload(_update(attachment_replay)) == attachment
    assert _update(complete) == {"sessionUpdate": "history_complete"}
    assert upload_response["result"]["session"]["_meta"] == {}
    assert load_response["result"]["_meta"] == {}
    assert download_response["result"]["attachment"] == attachment
    assert download_response["result"]["encoding"] == "base64"
    assert base64.b64decode(download_response["result"]["data"]) == b"\x89PNG\r\n\x1a\nimage"


@pytest.mark.asyncio
async def test_manager_upload_file_accepts_websocket_messages_over_one_mebibyte(tmp_path: Path) -> None:
    data = b"x" * (1024 * 1024 + 1)
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
                        name="large.txt",
                        mimeType="text/plain",
                        data=base64.b64encode(data).decode("ascii"),
                    ),
                )
            )
            upload_response, upload_updates = await _recv_response_with_updates(websocket)
            update = upload_updates[1]

    attachment = upload_response["result"]["attachment"]
    assert attachment["name"] == "large.txt"
    assert attachment["mimeType"] == "text/plain"
    assert attachment["size"] == len(data)
    assert _attachment_payload(_update(update)) == attachment


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
            upload_response, _upload_updates = await _recv_response_with_updates(websocket)
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
            upload_response, upload_updates = await _recv_response_with_updates(websocket)
            update = upload_updates[1]

    attachment = upload_response["result"]["attachment"]
    assert attachment["name"] == "attachment.txt"
    assert attachment["mimeType"] == "text/plain"
    assert attachment["path"].startswith("/attachments/att_")
    assert attachment["path"].endswith("-attachment.txt")
    filename = attachment["path"].removeprefix("/attachments/")
    assert (tmp_path / "sessions" / session_id / "attachments" / filename).read_text() == "clipboard text"
    update_payload = _update(update)
    assert update_payload["sessionUpdate"] == "attachment_added"
    assert _attachment_payload(update_payload) == attachment


@pytest.mark.asyncio
async def test_manager_upload_file_uses_unique_ids_for_duplicate_bytes(tmp_path: Path) -> None:
    data = base64.b64encode(b"same bytes").decode("ascii")
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
                        name="first.txt",
                        mimeType="text/plain",
                        data=data,
                    ),
                )
            )
            first_response, _first_updates = await _recv_response_with_updates(websocket)
            first_attachment = first_response["result"]["attachment"]

            await websocket.send(
                jsonrpc_request(
                    4,
                    "session/upload_file",
                    _scope_params(
                        "scope-a",
                        sessionId=session_id,
                        name="second.txt",
                        mimeType="text/plain",
                        data=data,
                    ),
                )
            )
            second_response, _second_updates = await _recv_response_with_updates(websocket)
            second_attachment = second_response["result"]["attachment"]

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/download_attachment",
                    _scope_params("scope-a", sessionId=session_id, attachmentId=first_attachment["id"]),
                )
            )
            first_download = parse_jsonrpc_message(await websocket.recv())
            await websocket.send(
                jsonrpc_request(
                    6,
                    "session/download_attachment",
                    _scope_params("scope-a", sessionId=session_id, attachmentId=second_attachment["id"]),
                )
            )
            second_download = parse_jsonrpc_message(await websocket.recv())

    assert first_attachment["id"] != second_attachment["id"]
    assert first_attachment["sha256"] == second_attachment["sha256"]
    assert first_attachment["name"] == "first.txt"
    assert second_attachment["name"] == "second.txt"
    assert first_download["result"]["attachment"] == first_attachment
    assert second_download["result"]["attachment"] == second_attachment


@pytest.mark.asyncio
async def test_manager_upload_file_accepts_pdf_and_instructs_text_extraction(tmp_path: Path) -> None:
    store = create_session_store(tmp_path)
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
    store = create_session_store(tmp_path)
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
    store = create_session_store(tmp_path)
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
    store.append_messages(
        scope_id="scope-a",
        session_id=created.record.session_id,
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
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(7)]
            response = parse_jsonrpc_message(await websocket.recv())

    replay_updates = [message["params"]["update"] for message in replay]
    assert _normalized_updates(replay_updates) == [
        {"sessionUpdate": "history_reset"},
        {"sessionUpdate": "message_added", "messageId": "message-0", "message": {"role": "system"}},
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "user", "content": "Read smoke.txt"},
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "shell_execute", "arguments": '{"command": "cat smoke.txt"}'},
                        "type": "function",
                    }
                ],
            },
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-3",
            "message": {
                "role": "tool",
                "content": "smoke output",
                "name": "shell_execute",
                "tool_call_id": "call-1",
            },
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-4",
            "message": {"role": "assistant", "content": "Done"},
        },
        {"sessionUpdate": "history_complete"},
    ]
    assert response["result"]["sessionId"] == created.record.session_id


@pytest.mark.asyncio
async def test_manager_load_replays_load_image_tool_message_content(tmp_path: Path) -> None:
    store = create_session_store(tmp_path)
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
    created = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])
    tool_call = ToolCall(
        id="call-1",
        function=FunctionCall(name="load_image", arguments='{"path": "/attachments/att_1-meal.png"}'),
    )
    store.append_messages(
        scope_id="scope-a",
        session_id=created.record.session_id,
        messages=[
            UserMessage(content="What is in this image?"),
            AssistantMessage(tool_calls=[tool_call]),
            ToolMessage(
                tool_call_id="call-1",
                name="load_image",
                content=[
                    {"type": "text", "text": "loaded image"},
                    IMAGE_BLOCK,
                ],
            ),
        ],
    )

    async with start_manager_server(service=service) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(
                jsonrpc_request(2, "session/load", _scope_params("scope-a", sessionId=created.record.session_id))
            )
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(6)]
            response = parse_jsonrpc_message(await websocket.recv())

    replay_updates = [message["params"]["update"] for message in replay]
    assert _normalized_updates(replay_updates) == [
        {"sessionUpdate": "history_reset"},
        {"sessionUpdate": "message_added", "messageId": "message-0", "message": {"role": "system"}},
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "user", "content": "What is in this image?"},
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "load_image", "arguments": '{"path": "/attachments/att_1-meal.png"}'},
                        "type": "function",
                    }
                ],
            },
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-3",
            "message": {
                "role": "tool",
                "name": "load_image",
                "tool_call_id": "call-1",
                "content": [
                    {"type": "text", "text": "loaded image"},
                    IMAGE_BLOCK,
                ],
            },
        },
        {"sessionUpdate": "history_complete"},
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
            rename_response, rename_updates = await _recv_response_with_updates(websocket)
            await websocket.send(jsonrpc_request(4, "session/list", _scope_params("scope-a")))
            list_response = parse_jsonrpc_message(await websocket.recv())

    assert rename_response["result"]["sessionId"] == session_id
    assert rename_response["result"]["title"] == "Renamed"
    assert _update(rename_updates[0])["sessionUpdate"] == "session_updated"
    assert _update(rename_updates[0])["session"]["title"] == "Renamed"
    assert list_response["result"]["sessions"][0]["title"] == "Renamed"


@pytest.mark.asyncio
async def test_manager_deletes_session_and_workspace(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(jsonrpc_request(3, "session/delete", _scope_params("scope-a", sessionId=session_id)))
            delete_response = parse_jsonrpc_message(await websocket.recv())
            await websocket.send(jsonrpc_request(4, "session/list", _scope_params("scope-a")))
            list_response = parse_jsonrpc_message(await websocket.recv())

    assert delete_response.get("result") is None
    assert list_response["result"]["sessions"] == []
    # Workspace and attachments directories are removed.
    assert not (tmp_path / "sessions" / session_id).exists()


@pytest.mark.asyncio
async def test_manager_delete_session_rejects_cross_scope(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")

            await websocket.send(jsonrpc_request(3, "session/delete", _scope_params("scope-b", sessionId=session_id)))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert "not found" in response["error"]["message"].lower()


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
            set_model_response = await _recv_response(websocket)

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("Do it")]),
                ),
            )
            prompt_response, live_updates = await _recv_response_with_updates(websocket)

            await websocket.send(jsonrpc_request(6, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(5)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    assert set_model_response["result"]["_meta"]["model"] == "alternate-model"
    live_payloads = [_update(message) for message in live_updates]
    live_message_payloads = [update for update in live_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _normalized_updates(live_message_payloads[:3]) == [
        {
            "sessionUpdate": "message_added",
            "messageId": "message-0",
            "message": {"role": "user", "content": "Do it"},
        },
        {
            "sessionUpdate": "message_delta",
            "messageId": "message-1",
            "appendText": "done",
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "assistant", "content": "done"},
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
            "message": {"role": "user", "content": "Do it"},
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {"role": "assistant", "content": "done"},
        },
        {"sessionUpdate": "history_complete"},
    ]
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
            prompt_response, live_updates = await _recv_response_with_updates(websocket)

            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(5)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    live_payloads = [_update(message) for message in live_updates]
    live_message_payloads = [update for update in live_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _normalized_updates(live_message_payloads[:3]) == [
        {
            "sessionUpdate": "message_added",
            "messageId": "message-0",
            "message": {"role": "user", "content": "Do it"},
        },
        {
            "sessionUpdate": "message_delta",
            "messageId": "message-1",
            "appendText": "done",
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "assistant", "content": "done"},
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
            "message": {"role": "user", "content": "Do it"},
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {"role": "assistant", "content": "done"},
        },
        {"sessionUpdate": "history_complete"},
    ]
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
async def test_manager_uses_prompt_worker_env_for_one_run(tmp_path: Path) -> None:
    worker = FakeWorkerRunner(response_text="done")
    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(
                websocket,
                scope_id="scope-a",
                metadata={
                    "workerEnv": {
                        "APPS_API_BASE_URL": "http://session-api",
                        "APPS_API_TOKEN": "stale-session-token",
                    },
                },
            )
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)
            params = _scope_params(
                "scope-a",
                sessionId=session_id,
                prompt=[text_block("Do it")],
            )
            params["_meta"]["workerEnv"] = {
                "APPS_API_TOKEN": "fresh-prompt-token",
                "PROMPT_ONLY": "prompt-value",
            }

            await websocket.send(jsonrpc_request(4, "session/prompt", params))
            response, live_updates = await _recv_response_with_updates(websocket)
            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/prompt",
                    _scope_params(
                        "scope-a",
                        sessionId=session_id,
                        prompt=[text_block("Do it again")],
                    ),
                )
            )
            second_response, _second_live_updates = await _recv_response_with_updates(websocket)

    live_payloads = [_update(message) for message in live_updates]
    live_message_payloads = [update for update in live_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _normalized_updates(live_message_payloads[:3]) == [
        {
            "sessionUpdate": "message_added",
            "messageId": "message-0",
            "message": {"role": "user", "content": "Do it"},
        },
        {
            "sessionUpdate": "message_delta",
            "messageId": "message-1",
            "appendText": "done",
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "assistant", "content": "done"},
        },
    ]
    assert response["result"] == {"stopReason": "end_turn"}
    assert second_response["result"] == {"stopReason": "end_turn"}
    assert worker.prompts is not None
    assert worker.prompts[0].worker_env == {
        "APPS_API_BASE_URL": "http://session-api",
        "APPS_API_TOKEN": "fresh-prompt-token",
        "PROMPT_ONLY": "prompt-value",
    }
    assert worker.prompts[1].worker_env == {
        "APPS_API_BASE_URL": "http://session-api",
        "APPS_API_TOKEN": "stale-session-token",
    }


@pytest.mark.asyncio
async def test_manager_rejects_prompt_scoped_skills(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            session_id = await _new_session(websocket, scope_id="scope-a")
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)
            params = _scope_params(
                "scope-a",
                sessionId=session_id,
                prompt=[text_block("Do it")],
            )
            params["_meta"]["skills"] = [
                {
                    "files": {
                        "SKILL.md": "---\nname: prompt-skill\ndescription: Use prompt-provided tools.\n---\n",
                    },
                },
            ]

            await websocket.send(jsonrpc_request(4, "session/prompt", params))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "session/prompt does not accept _meta.skills; pass skills to session/new."


@pytest.mark.asyncio
async def test_manager_validates_injected_skill_paths(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            params = _scope_params("scope-a")
            params["_meta"]["skills"] = [
                {
                    "files": {
                        "SKILL.md": "---\nname: apps-api\ndescription: Use apps REST APIs.\n---\n",
                        "../escape.md": "bad",
                    },
                },
            ]

            await websocket.send(jsonrpc_request(2, "session/new", params))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Injected skill apps-api has an invalid file path: '../escape.md'."


@pytest.mark.parametrize(
    ("skill_file", "message"),
    [
        ("---\ndescription: Missing name.\n---\n", "Invalid injected skill name: None."),
        ("---\nname: missing-description\n---\n", "Injected skill missing-description requires a description."),
        ("---\nname: [\n---\n", "Invalid injected skill SKILL.md frontmatter."),
    ],
)
@pytest.mark.asyncio
async def test_manager_validates_injected_skill_frontmatter(tmp_path: Path, skill_file: str, message: str) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            params = _scope_params("scope-a")
            params["_meta"]["skills"] = [{"files": {"SKILL.md": skill_file}}]

            await websocket.send(jsonrpc_request(2, "session/new", params))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == message


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
            first_updates: list[dict[str, Any]] = []
            saw_delta = False
            while not saw_delta:
                message = parse_jsonrpc_message(await websocket.recv())
                first_updates.append(message)
                saw_delta = _update(message).get("sessionUpdate") == "message_delta"
            await asyncio.wait_for(started.wait(), timeout=1)

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/set_model",
                    _scope_params("scope-a", sessionId=session_id, model="alternate-model"),
                ),
            )
            busy_response = await _recv_response(websocket)
            release.set()
            first_response = await _recv_response(websocket)

    first_payloads = [_update(message) for message in first_updates]
    first_message_payloads = [update for update in first_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _message_payload(first_message_payloads[0]) == {"role": "user", "content": "first"}
    assert first_message_payloads[1]["appendText"] == "fake response"
    assert busy_response["error"]["code"] == -32000
    assert busy_response["error"]["message"] == "Cannot change model while session has an active prompt."
    assert first_response["result"] == {"stopReason": "end_turn"}


@pytest.mark.asyncio
async def test_manager_logs_prompt_request_errors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class FailingWorker:
        async def run_prompt(
            self,
            *,
            prompt: WorkerPrompt,
            on_update: Callable[[SessionUpdate], Awaitable[None]],
        ) -> WorkerRunResult:
            del prompt
            await on_update(MessageDeltaUpdate(message_id="msg_partial", append_text="partial response"))
            raise RuntimeError("provider returned JSON instead of event-stream")

        async def cancel(self, *, session_id: str) -> None:
            del session_id

    caplog.set_level(logging.ERROR, logger="coding_assistant.manager.server")

    async with _manager_endpoint(tmp_path=tmp_path, worker=FailingWorker()) as endpoint:
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
            response, updates = await _recv_response_with_updates(websocket)
            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            _, replay = await _recv_response_with_updates(websocket)

    assert any(
        _update(update).get("sessionUpdate") == "message_added"
        and _message_payload(_update(update)) == {"role": "user", "content": "Do it"}
        for update in updates
    )
    assert any(
        _update(update).get("sessionUpdate") == "message_delta"
        and _update(update).get("appendText") == "partial response"
        for update in updates
    )
    assert any(
        _update(update).get("sessionUpdate") == "run_updated"
        and isinstance(_update(update).get("run"), dict)
        and _update(update)["run"].get("status") == "failed"
        for update in updates
    )
    assert not any(_update(update).get("sessionUpdate") == "history_reset" for update in updates)
    assert _normalized_updates([_update(update) for update in replay]) == [
        {"sessionUpdate": "history_reset"},
        {"sessionUpdate": "message_added", "messageId": "message-0", "message": {"role": "system"}},
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "user", "content": "Do it"},
        },
        {"sessionUpdate": "history_complete"},
    ]
    assert response["error"]["message"] == "provider returned JSON instead of event-stream"
    assert f"Manager prompt request failed for session {session_id}." in caplog.text
    assert "provider returned JSON instead of event-stream" in caplog.text
    assert "Traceback" in caplog.text


@pytest.mark.asyncio
async def test_manager_logs_method_errors(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="coding_assistant.manager.server")

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

    assert response["error"]["message"] == "Unsupported attachment MIME type: application/zip."
    assert "Manager method session/upload_file failed." in caplog.text
    assert "Unsupported attachment MIME type: application/zip." in caplog.text


@pytest.mark.asyncio
async def test_manager_prompt_streams_update_and_persists_messages(tmp_path: Path) -> None:
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
            response, live_updates = await _recv_response_with_updates(websocket)

            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay_messages = [parse_jsonrpc_message(await websocket.recv()) for _ in range(5)]
            parse_jsonrpc_message(await websocket.recv())

    live_payloads = [_update(message) for message in live_updates]
    live_message_payloads = [update for update in live_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _normalized_updates(live_message_payloads[:3]) == [
        {
            "sessionUpdate": "message_added",
            "messageId": "message-0",
            "message": {"role": "user", "content": "Do it"},
        },
        {
            "sessionUpdate": "message_delta",
            "messageId": "message-1",
            "appendText": "done",
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "assistant", "content": "done"},
        },
    ]
    assert response["result"] == {"stopReason": "end_turn"}
    replay_payloads = [_update(message) for message in replay_messages]
    assert _normalized_updates(replay_payloads) == [
        {"sessionUpdate": "history_reset"},
        {"sessionUpdate": "message_added", "messageId": "message-0", "message": {"role": "system"}},
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {"role": "user", "content": "Do it"},
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {"role": "assistant", "content": "done"},
        },
        {"sessionUpdate": "history_complete"},
    ]


@pytest.mark.asyncio
async def test_manager_prompt_survives_client_disconnect_and_reconnect(tmp_path: Path) -> None:
    class PausingWorker:
        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def run_prompt(
            self,
            *,
            prompt: WorkerPrompt,
            on_update: Callable[[SessionUpdate], Awaitable[None]],
        ) -> WorkerRunResult:
            del prompt
            await on_update(MessageDeltaUpdate(message_id="msg_worker", append_text="done"))
            await self.release.wait()
            assistant_message = AssistantMessage(content="done")
            await on_update(MessageAddedUpdate(message_id="msg_worker", message=assistant_message))
            return WorkerRunResult(stop_reason="end_turn")

        async def cancel(self, *, session_id: str) -> None:
            del session_id

    worker = PausingWorker()

    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as first:
            await _initialize(first)
            session_id = await _new_session(first, scope_id="scope-a")
            await _set_model(first, scope_id="scope-a", session_id=session_id, request_id=3)

            await first.send(
                jsonrpc_request(
                    4,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("Do it")]),
                ),
            )
            first_updates: list[dict[str, Any]] = []
            saw_delta = False
            while not saw_delta:
                message = parse_jsonrpc_message(await first.recv())
                first_updates.append(message)
                update = _update(message)
                saw_delta = update.get("sessionUpdate") == "message_delta"

        assert any(_update(message).get("sessionUpdate") == "run_updated" for message in first_updates)
        live_message_id = next(
            _update(message)["messageId"]
            for message in first_updates
            if _update(message).get("sessionUpdate") == "message_delta"
        )

        async with connect(endpoint) as second:
            await _initialize(second)
            await second.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            load_response, replay = await _recv_response_with_updates(second)
            worker.release.set()

            completed_run: dict[str, Any] | None = None
            completed_message_id: str | None = None
            while completed_run is None:
                message = parse_jsonrpc_message(await second.recv())
                update = _update(message)
                if update.get("sessionUpdate") == "message_added" and _message_payload(update) == {
                    "role": "assistant",
                    "content": "done",
                }:
                    message_payload = update.get("message")
                    if isinstance(message_payload, dict):
                        message_id = message_payload.get("id")
                        if isinstance(message_id, str):
                            completed_message_id = message_id
                if update.get("sessionUpdate") == "run_updated":
                    run = update.get("run")
                    if isinstance(run, dict) and run.get("status") == "completed":
                        completed_run = run

            await second.send(jsonrpc_request(6, "session/load", _scope_params("scope-a", sessionId=session_id)))
            final_load_response, final_replay = await _recv_response_with_updates(second)

    replay_payloads = [_update(message) for message in replay]
    assert {
        tuple(item.get("message", {}).items())
        for item in _normalized_updates(replay_payloads)
        if item.get("sessionUpdate") == "message_added"
    } >= {
        tuple({"role": "user", "content": "Do it"}.items()),
    }
    assert any(
        update.get("sessionUpdate") == "message_delta"
        and update.get("messageId") == live_message_id
        and update.get("appendText") == "done"
        for update in replay_payloads
    )
    assert any(
        update.get("sessionUpdate") == "run_updated"
        and isinstance(update.get("run"), dict)
        and update["run"].get("status") == "running"
        for update in replay_payloads
    )
    assert load_response["result"]["sessionId"] == session_id
    assert completed_message_id == live_message_id
    assert completed_run["stopReason"] == "end_turn"
    final_payloads = [_update(message) for message in final_replay]
    assert {
        tuple(item.get("message", {}).items())
        for item in _normalized_updates(final_payloads)
        if item.get("sessionUpdate") == "message_added"
    } >= {
        tuple({"role": "user", "content": "Do it"}.items()),
        tuple({"role": "assistant", "content": "done"}.items()),
    }


@pytest.mark.asyncio
async def test_manager_replaces_a_superseded_streaming_attempt(tmp_path: Path) -> None:
    class RetryingWorker:
        async def run_prompt(
            self,
            *,
            prompt: WorkerPrompt,
            on_update: Callable[[SessionUpdate], Awaitable[None]],
        ) -> WorkerRunResult:
            del prompt
            await on_update(MessageDeltaUpdate(message_id="attempt-1", append_text="discarded"))
            await on_update(MessageDeltaUpdate(message_id="attempt-2", append_text="kept"))
            assistant_message = AssistantMessage(content="kept")
            await on_update(MessageAddedUpdate(message_id="attempt-2", message=assistant_message))
            return WorkerRunResult(stop_reason="end_turn")

        async def cancel(self, *, session_id: str) -> None:
            del session_id

    async with _manager_endpoint(tmp_path=tmp_path, worker=RetryingWorker()) as endpoint:
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
            response, _ = await _recv_response_with_updates(websocket)
            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            _, replay = await _recv_response_with_updates(websocket)

    assert response["result"] == {"stopReason": "end_turn"}
    replayed_messages = [
        _message_payload(update)
        for update in (_update(message) for message in replay)
        if update.get("sessionUpdate") == "message_added"
    ]
    assert replayed_messages == [
        {"role": "system"},
        {"role": "user", "content": "Do it"},
        {"role": "assistant", "content": "kept"},
    ]


@pytest.mark.asyncio
async def test_manager_persists_live_tool_messages(tmp_path: Path) -> None:
    class ToolStreamingWorker:
        async def run_prompt(
            self,
            *,
            prompt: WorkerPrompt,
            on_update: Callable[[SessionUpdate], Awaitable[None]],
        ) -> WorkerRunResult:
            del prompt
            tool_call = ToolCall(
                id="call-1",
                function=FunctionCall(name="load_image", arguments='{"path": "/attachments/att_1-meal.png"}'),
            )
            tool_call_message = AssistantMessage(tool_calls=[tool_call])
            tool_message = ToolMessage(
                tool_call_id="call-1",
                name="load_image",
                content=[
                    {"type": "text", "text": "loaded image"},
                    IMAGE_BLOCK,
                ],
            )
            final_message = AssistantMessage(content="done")
            await on_update(MessageAddedUpdate(message_id="tool_calls", message=tool_call_message))
            await on_update(MessageAddedUpdate(message_id="tool_result", message=tool_message))
            await on_update(MessageDeltaUpdate(message_id="msg_worker", append_text="done"))
            await on_update(MessageAddedUpdate(message_id="msg_worker", message=final_message))
            return WorkerRunResult(stop_reason="end_turn")

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
            response, live_messages = await _recv_response_with_updates(websocket)

    live_payloads = [_update(message) for message in live_messages]
    live_message_payloads = [update for update in live_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _normalized_updates(live_message_payloads[:5]) == [
        {
            "sessionUpdate": "message_added",
            "messageId": "message-0",
            "message": {"role": "user", "content": "Use tool"},
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-1",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "load_image", "arguments": '{"path": "/attachments/att_1-meal.png"}'},
                        "type": "function",
                    }
                ],
            },
        },
        {
            "sessionUpdate": "message_added",
            "messageId": "message-2",
            "message": {
                "role": "tool",
                "name": "load_image",
                "tool_call_id": "call-1",
                "content": [
                    {"type": "text", "text": "loaded image"},
                    IMAGE_BLOCK,
                ],
            },
        },
        {"sessionUpdate": "message_delta", "messageId": "message-3", "appendText": "done"},
        {
            "sessionUpdate": "message_added",
            "messageId": "message-3",
            "message": {"role": "assistant", "content": "done"},
        },
    ]
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
            first_updates: list[dict[str, Any]] = []
            saw_delta = False
            while not saw_delta:
                message = parse_jsonrpc_message(await websocket.recv())
                first_updates.append(message)
                saw_delta = _update(message).get("sessionUpdate") == "message_delta"
            await asyncio.wait_for(started.wait(), timeout=1)

            await websocket.send(
                jsonrpc_request(
                    5,
                    "session/prompt",
                    _scope_params("scope-a", sessionId=session_id, prompt=[text_block("second")]),
                ),
            )
            busy_response = await _recv_response(websocket)
            release.set()
            first_response = await _recv_response(websocket)

    first_payloads = [_update(message) for message in first_updates]
    first_message_payloads = [update for update in first_payloads if update.get("sessionUpdate") != "run_updated"]
    assert _message_payload(first_message_payloads[0]) == {"role": "user", "content": "first"}
    assert first_message_payloads[1]["appendText"] == "fake response"
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
            first_response = await _recv_response(first)
            second_response = await _recv_response(second)

    assert first_updates[0]["params"]["sessionId"] == first_session_id
    assert second_updates[0]["params"]["sessionId"] == second_session_id
    assert first_response["result"] == {"stopReason": "end_turn"}
    assert second_response["result"] == {"stopReason": "end_turn"}
