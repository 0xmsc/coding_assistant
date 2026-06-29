from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from coding_assistant.llm.types import AssistantMessage, FunctionCall, SystemMessage, ToolCall, ToolMessage, UserMessage
from coding_assistant.manager.server import start_manager_server
from coding_assistant.manager.service import MAX_ATTACHMENT_BYTES, ManagerError, ManagerService
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
    worker: FakeWorkerRunner | None = None,
    auth_secret: str | None = None,
) -> AsyncIterator[str]:
    store = SessionStore(
        database_path=tmp_path / "sessions.sqlite",
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    service = ManagerService(
        store=store,
        worker_runner=worker or FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
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
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_failing_model_lister,
    )

    async with start_manager_server(service=service, initial_messages=[SystemMessage(content="system")]) as server:
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
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
    )
    service = ManagerService(
        store=store,
        worker_runner=FakeWorkerRunner(),
        model_lister=_test_model_lister,
    )
    created = store.create_session(scope_id="scope-a", messages=[SystemMessage(content="system")])

    async with start_manager_server(service=service, initial_messages=[SystemMessage(content="system")]) as server:
        async with connect(server.endpoint) as websocket:
            await _initialize(websocket)
            await websocket.send(jsonrpc_request(2, "session/list", _scope_params("scope-a")))
            list_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(
                jsonrpc_request(3, "session/load", _scope_params("scope-a", sessionId=created.record.session_id))
            )
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
            response = parse_jsonrpc_message(await websocket.recv())

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
            replay = parse_jsonrpc_message(await websocket.recv())
            load_response = parse_jsonrpc_message(await websocket.recv())

    attachment = upload_response["result"]["attachment"]
    assert attachment["name"] == "Meal-Photo.PNG"
    assert attachment["mimeType"] == "image/png"
    assert attachment["size"] == 13
    assert attachment["path"].startswith("attachments/att_")
    assert attachment["path"].endswith("-Meal-Photo.PNG")
    assert (tmp_path / "workspaces" / session_id / attachment["path"]).read_bytes() == b"\x89PNG\r\n\x1a\nimage"

    expected_text = (
        f"Attached file `Meal-Photo.PNG` as `{attachment['path']}` (image/png, 13 bytes). "
        f'Use `load_file("{attachment["path"]}")` before reasoning from this file.'
    )
    assert update["params"]["update"] == {
        "sessionUpdate": "user_message_chunk",
        "content": {"type": "text", "text": expected_text},
    }
    assert replay["params"]["update"] == update["params"]["update"]
    assert upload_response["result"]["session"]["_meta"]["version"] == 1
    assert load_response["result"]["_meta"]["version"] == 1


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
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
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
        workspaces=WorkspacePaths(root=tmp_path / "workspaces"),
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
            update = parse_jsonrpc_message(await websocket.recv())
            prompt_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(6, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(2)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    assert set_model_response["result"]["_meta"]["model"] == "alternate-model"
    assert update["params"]["update"]["content"]["text"] == "done"
    assert prompt_response["result"] == {"stopReason": "end_turn"}
    assert [message["params"]["update"]["content"]["text"] for message in replay] == ["Do it", "done"]
    assert load_response["result"]["_meta"]["model"] == "alternate-model"
    assert worker.prompts is not None
    assert worker.prompts[0].model == "alternate-model"


@pytest.mark.asyncio
async def test_manager_stores_session_worker_env_privately_and_passes_prompt_skills(tmp_path: Path) -> None:
    worker = FakeWorkerRunner(response_text="done")
    async with _manager_endpoint(tmp_path=tmp_path, worker=worker) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            new_session_params = _scope_params("scope-a")
            new_session_params["_meta"]["workerEnv"] = {
                "APPS_API_BASE_URL": "http://apps-api",
                "APPS_API_TOKEN": "secret-token",
            }
            await websocket.send(jsonrpc_request(2, "session/new", new_session_params))
            new_session_response = parse_jsonrpc_message(await websocket.recv())
            session_id = new_session_response["result"]["sessionId"]
            await _set_model(websocket, scope_id="scope-a", session_id=session_id, request_id=3)
            params = _scope_params(
                "scope-a",
                sessionId=session_id,
                prompt=[text_block("Do it")],
            )
            params["_meta"]["capabilities"] = {
                "skills": [
                    {
                        "name": "apps-api",
                        "description": "Use apps REST APIs.",
                        "files": {
                            "SKILL.md": "---\nname: apps-api\ndescription: Use apps REST APIs.\n---\n",
                            "references/calories.md": "calories",
                        },
                    },
                ],
            }

            await websocket.send(jsonrpc_request(4, "session/prompt", params))
            update = parse_jsonrpc_message(await websocket.recv())
            prompt_response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay = [parse_jsonrpc_message(await websocket.recv()) for _ in range(2)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    assert update["params"]["update"]["content"]["text"] == "done"
    assert prompt_response["result"] == {"stopReason": "end_turn"}
    assert [message["params"]["update"]["content"]["text"] for message in replay] == ["Do it", "done"]
    assert "capabilities" not in load_response["result"]["_meta"]
    assert "workerEnv" not in load_response["result"]["_meta"]
    assert worker.prompts is not None
    assert worker.prompts[0].worker_env == {
        "APPS_API_BASE_URL": "http://apps-api",
        "APPS_API_TOKEN": "secret-token",
    }
    capabilities = worker.prompts[0].capabilities
    assert len(capabilities.skills) == 1
    assert capabilities.skills[0].name == "apps-api"
    assert capabilities.skills[0].files["references/calories.md"] == "calories"


@pytest.mark.asyncio
async def test_manager_rejects_capabilities_on_non_prompt_methods(tmp_path: Path) -> None:
    async with _manager_endpoint(tmp_path=tmp_path) as endpoint:
        async with connect(endpoint) as websocket:
            await _initialize(websocket)
            params = _scope_params("scope-a")
            params["_meta"]["capabilities"] = {"workerEnv": {"APPS_API_TOKEN": "secret"}}

            await websocket.send(jsonrpc_request(2, "session/list", params))
            response = parse_jsonrpc_message(await websocket.recv())

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "_meta.capabilities is only accepted on session/prompt."


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
            update = parse_jsonrpc_message(await websocket.recv())
            response = parse_jsonrpc_message(await websocket.recv())

    assert update["params"]["update"]["content"]["text"] == "done"
    assert response["result"] == {"stopReason": "end_turn"}
    assert worker.prompts is not None
    assert worker.prompts[0].worker_env == {}


@pytest.mark.asyncio
async def test_manager_validates_injected_skill_paths(tmp_path: Path) -> None:
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
            params["_meta"]["capabilities"] = {
                "skills": [
                    {
                        "name": "apps-api",
                        "description": "Use apps REST APIs.",
                        "files": {
                            "SKILL.md": "skill",
                            "../escape.md": "bad",
                        },
                    },
                ],
            }

            await websocket.send(jsonrpc_request(4, "session/prompt", params))
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
            first_update = parse_jsonrpc_message(await websocket.recv())
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

    assert first_update["params"]["update"]["content"]["text"] == "fake response"
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
            update = parse_jsonrpc_message(await websocket.recv())
            response = parse_jsonrpc_message(await websocket.recv())

            await websocket.send(jsonrpc_request(5, "session/load", _scope_params("scope-a", sessionId=session_id)))
            replay_messages = [parse_jsonrpc_message(await websocket.recv()) for _ in range(2)]
            load_response = parse_jsonrpc_message(await websocket.recv())

    assert update["params"]["update"]["content"]["text"] == "done"
    assert response["result"] == {"stopReason": "end_turn"}
    replay_texts = [message["params"]["update"]["content"]["text"] for message in replay_messages]
    assert replay_texts == ["Do it", "done"]
    assert load_response["result"]["_meta"]["version"] == 1


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
            first_update = parse_jsonrpc_message(await websocket.recv())
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
            first_update = parse_jsonrpc_message(await first.recv())
            second_update = parse_jsonrpc_message(await second.recv())
            release.set()
            first_response = parse_jsonrpc_message(await first.recv())
            second_response = parse_jsonrpc_message(await second.recv())

    assert first_update["params"]["sessionId"] == first_session_id
    assert second_update["params"]["sessionId"] == second_session_id
    assert first_response["result"] == {"stopReason": "end_turn"}
    assert second_response["result"] == {"stopReason": "end_turn"}
