from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import ClientConnection, connect

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.manager.docker_worker import (
    DockerCommandResult,
    DockerWorkerConfig,
    DockerWorkerError,
    DockerWorkerRunner,
    TOOL_ENV_KEYS_ENV,
    _docker_run_args,
    _run_command,
)
from coding_assistant.manager.service import WorkerPrompt
from coding_assistant.remote.jsonrpc import ACP_PROTOCOL_VERSION, jsonrpc_request, parse_jsonrpc_message, text_block


def _docker(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def _require_docker() -> None:
    result = _docker(["version", "--format", "{{.Server.Version}}"], timeout=10)
    if result.returncode != 0:
        pytest.skip(f"Docker daemon is not available: {result.stderr.strip() or result.stdout.strip()}")


def _remove_container(name: str) -> None:
    _docker(["rm", "-f", name], timeout=20)


def _remove_network(name: str) -> None:
    _docker(["network", "rm", name], timeout=20)


def _remove_image(name: str) -> None:
    _docker(["image", "rm", "-f", name], timeout=30)


def _docker_socket_group_id() -> str:
    return str(Path("/var/run/docker.sock").stat().st_gid)


def _published_port(container_name: str, container_port: int) -> str:
    result = _docker(["port", container_name, str(container_port)], timeout=20)
    assert result.returncode == 0, result.stderr or result.stdout
    first_mapping = result.stdout.strip().splitlines()[0]
    return first_mapping.rsplit(":", maxsplit=1)[1]


async def _wait_for_manager_endpoint(endpoint: str, *, auth_token: str) -> None:
    deadline = asyncio.get_running_loop().time() + 30
    last_error = "manager did not respond"
    while asyncio.get_running_loop().time() < deadline:
        try:
            async with connect(endpoint, additional_headers={"Authorization": f"Bearer {auth_token}"}) as websocket:
                await _initialize(websocket)
                return
        except Exception as exc:
            last_error = str(exc)
            await asyncio.sleep(0.2)
    raise AssertionError(f"Manager endpoint {endpoint} did not become ready: {last_error}")


@asynccontextmanager
async def _connect_initialized(endpoint: str, *, auth_token: str) -> AsyncIterator[ClientConnection]:
    async with connect(endpoint, additional_headers={"Authorization": f"Bearer {auth_token}"}) as websocket:
        await _initialize(websocket)
        yield websocket


async def _initialize(websocket: ClientConnection) -> dict[str, Any]:
    await websocket.send(
        jsonrpc_request(
            1,
            "initialize",
            {
                "protocolVersion": ACP_PROTOCOL_VERSION,
                "clientCapabilities": {},
                "clientInfo": {"name": "docker-smoke", "title": "Docker Smoke", "version": "0.0.0"},
            },
        ),
    )
    response = parse_jsonrpc_message(await websocket.recv())
    assert "result" in response, response
    return response


def _scope_params(session_id: str | None = None, **params: Any) -> dict[str, Any]:
    result = {
        "_meta": {
            "scopeId": "scope-a",
        },
        **params,
    }
    if session_id is not None:
        result["sessionId"] = session_id
    return result


async def _receive_response(websocket: ClientConnection, *, request_id: int) -> dict[str, Any]:
    while True:
        message = parse_jsonrpc_message(await websocket.recv())
        if message.get("id") == request_id:
            assert "result" in message, message
            return message


async def _new_session(websocket: ClientConnection, *, request_id: int) -> str:
    await websocket.send(jsonrpc_request(request_id, "session/new", _scope_params()))
    response = await _receive_response(websocket, request_id=request_id)
    session_id = response["result"]["sessionId"]
    assert isinstance(session_id, str)
    return session_id


async def _prompt_session(websocket: ClientConnection, *, request_id: int, session_id: str) -> dict[str, Any]:
    await websocket.send(
        jsonrpc_request(
            request_id,
            "session/prompt",
            _scope_params(session_id, prompt=[text_block("read smoke.txt")]),
        ),
    )
    return await _receive_response(websocket, request_id=request_id)


async def _load_session(websocket: ClientConnection, *, request_id: int, session_id: str) -> list[str]:
    await websocket.send(jsonrpc_request(request_id, "session/load", _scope_params(session_id)))
    texts: list[str] = []
    while True:
        message = parse_jsonrpc_message(await websocket.recv())
        if message.get("id") == request_id:
            assert "result" in message, message
            return texts
        if message.get("method") != "session/update":
            continue
        update = message["params"]["update"]
        if update.get("sessionUpdate") == "message_added":
            message_payload = update.get("message")
            if not isinstance(message_payload, dict):
                continue
            if message_payload.get("role") == "assistant":
                content = message_payload.get("content")
                if isinstance(content, str):
                    texts.append(content)


def test_docker_run_args_mount_session_workspace_and_start_worker() -> None:
    config = DockerWorkerConfig(
        image="coding-assistant:test",
        network="assistant-net",
        environment={"OPENAI_API_KEY": "test-key"},
        instructions=("Be concise.",),
        skills_directories=("/skills",),
        extra_hosts=("host.docker.internal:host-gateway",),
    )

    args = _docker_run_args(
        config=config,
        container_name="coding-assistant-worker-sess",
        workspace="/data/sessions/sess/workspace",
        attachments="/data/sessions/sess/attachments",
        model="test-model",
    )

    assert args[:4] == ["docker", "run", "--detach", "--rm"]
    assert ["--name", "coding-assistant-worker-sess"] == args[4:6]
    assert "--security-opt" in args
    assert "no-new-privileges" in args
    assert "-v" in args
    assert "/data/sessions/sess/workspace:/workspace:rw" in args
    assert "/data/sessions/sess/attachments:/attachments:ro" in args
    assert "coding-assistant:test" in args
    assert "coding-assistant-worker" in args
    assert args[args.index("--network") + 1] == "assistant-net"
    assert args[args.index("--model") + 1] == "test-model"
    assert args[args.index("--host") + 1] == "0.0.0.0"
    assert args[args.index("--workspace") + 1] == "/workspace"
    assert "OPENAI_API_KEY=test-key" in args
    assert ["--add-host", "host.docker.internal:host-gateway"] == args[
        args.index("--add-host") : args.index("--add-host") + 2
    ]
    assert "Be concise." in args
    assert "/skills" in args


def test_docker_run_args_merges_session_environment() -> None:
    config = DockerWorkerConfig(
        image="coding-assistant:test",
        network="assistant-net",
        environment={"OPENAI_API_KEY": "test-key"},
        skills_directories=("/skills",),
    )

    args = _docker_run_args(
        config=config,
        container_name="coding-assistant-worker-sess",
        workspace="/data/sessions/sess/workspace",
        attachments="/data/sessions/sess/attachments",
        model="test-model",
        extra_environment={
            "APPS_API_BASE_URL": "http://apps-api",
            "APPS_API_TOKEN": "secret-token",
        },
    )

    assert "OPENAI_API_KEY=test-key" in args
    assert "APPS_API_BASE_URL=http://apps-api" in args
    assert "APPS_API_TOKEN=secret-token" in args
    assert f"{TOOL_ENV_KEYS_ENV}=APPS_API_BASE_URL,APPS_API_TOKEN" in args
    skills_index = args.index("--skills-directories")
    assert args[skills_index + 1 : skills_index + 2] == ["/skills"]


def test_docker_run_args_rejects_session_environment_collision() -> None:
    config = DockerWorkerConfig(
        image="coding-assistant:test",
        network="assistant-net",
        environment={"OPENAI_API_KEY": "test-key"},
    )

    with pytest.raises(
        DockerWorkerError,
        match="Session worker environment collides with manager environment: OPENAI_API_KEY",
    ):
        _docker_run_args(
            config=config,
            container_name="coding-assistant-worker-sess",
            workspace="/data/sessions/sess/workspace",
            attachments="/data/sessions/sess/attachments",
            model="test-model",
            extra_environment={"OPENAI_API_KEY": "attacker-key"},
        )


@pytest.mark.parametrize(
    ("base_environment", "extra_environment"),
    [
        ({TOOL_ENV_KEYS_ENV: "OPENAI_API_KEY"}, {}),
        ({}, {TOOL_ENV_KEYS_ENV: "APPS_API_TOKEN"}),
    ],
)
def test_docker_run_args_rejects_reserved_tool_env_key(
    base_environment: dict[str, str], extra_environment: dict[str, str]
) -> None:
    config = DockerWorkerConfig(
        image="coding-assistant:test",
        network="assistant-net",
        environment=base_environment,
    )

    with pytest.raises(
        DockerWorkerError,
        match=f"{TOOL_ENV_KEYS_ENV} is reserved for manager-provided worker tool env.",
    ):
        _docker_run_args(
            config=config,
            container_name="coding-assistant-worker-sess",
            workspace="/data/sessions/sess/workspace",
            attachments="/data/sessions/sess/attachments",
            model="test-model",
            extra_environment=extra_environment,
        )


def test_docker_run_args_maps_manager_session_paths_to_host_source() -> None:
    config = DockerWorkerConfig(
        image="coding-assistant:test",
        network="assistant-net",
        manager_session_root="/data/sessions",
        session_source_root="/host/coding-assistant/sessions",
    )

    args = _docker_run_args(
        config=config,
        container_name="coding-assistant-worker-sess",
        workspace="/data/sessions/sess/workspace",
        attachments="/data/sessions/sess/attachments",
        model="test-model",
    )

    assert "/host/coding-assistant/sessions/sess/workspace:/workspace:rw" in args
    assert "/host/coding-assistant/sessions/sess/attachments:/attachments:ro" in args


@pytest.mark.asyncio
async def test_docker_worker_runner_logs_container_output_before_removal(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    commands: list[list[str]] = []

    async def fake_run_command(args: Sequence[str]) -> DockerCommandResult:
        commands.append(list(args))
        return DockerCommandResult(stdout="worker stdout\n", stderr="worker stderr\n", returncode=0)

    monkeypatch.setattr("coding_assistant.manager.docker_worker._run_command", fake_run_command)
    caplog.set_level(logging.ERROR, logger="coding_assistant.manager.docker_worker")

    runner = DockerWorkerRunner(config=DockerWorkerConfig(image="image", network="network"))
    await runner._log_container_output("coding-assistant-worker-sess")

    assert commands == [["docker", "logs", "--tail", "200", "coding-assistant-worker-sess"]]
    assert "Worker container coding-assistant-worker-sess stdout before removal:\nworker stdout" in caplog.text
    assert "Worker container coding-assistant-worker-sess stderr before removal:\nworker stderr" in caplog.text


@pytest.mark.asyncio
@pytest.mark.slow
async def test_docker_manager_runs_two_sessions_with_shell_tool_calls(tmp_path: Path) -> None:
    _require_docker()
    suffix = uuid.uuid4().hex[:12]
    image = f"coding-assistant:test-{suffix}"
    network = f"coding-assistant-test-{suffix}"
    fake_openai = f"coding-assistant-fake-openai-{suffix}"
    manager = f"coding-assistant-manager-smoke-{suffix}"
    manager_auth_secret = f"manager-secret-{suffix}"
    session_id = f"sess-{suffix}"
    worker_container = f"coding-assistant-worker-{session_id}"
    fake_openai_script = tmp_path / "fake-openai-responses.json"
    fake_openai_script.write_text(
        """
[
  {
    "tool_calls": [
      {
        "id": "call_shell_first",
        "name": "shell_execute",
        "arguments": {"command": "cat smoke.txt"}
      }
    ]
  },
  {
    "content": "tool result: alpha from first workspace"
  },
  {
    "tool_calls": [
      {
        "id": "call_shell_second",
        "name": "shell_execute",
        "arguments": {"command": "cat smoke.txt"}
      }
    ]
  },
  {
    "content": "tool result: bravo from second workspace"
  }
]
""".strip(),
        encoding="utf-8",
    )

    try:
        build = _docker(["build", "-f", "docker/Dockerfile", "-t", image, "."], timeout=180)
        assert build.returncode == 0, build.stderr or build.stdout

        created_network = _docker(["network", "create", network], timeout=20)
        assert created_network.returncode == 0, created_network.stderr or created_network.stdout

        fake_started = _docker(
            [
                "run",
                "--detach",
                "--rm",
                "--name",
                fake_openai,
                "--network",
                network,
                "-e",
                f"CODING_ASSISTANT_FAKE_OPENAI_RESPONSES_FILE={fake_openai_script}",
                "-v",
                f"{fake_openai_script}:{fake_openai_script}:ro",
                image,
                "coding-assistant-fake-openai",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            timeout=30,
        )
        assert fake_started.returncode == 0, fake_started.stderr or fake_started.stdout

        image_user = _docker(["image", "inspect", image, "--format", "{{.Config.User}}"], timeout=20)
        assert image_user.returncode == 0, image_user.stderr or image_user.stdout
        assert image_user.stdout.strip(), image_user.stdout
        manager_started = _docker(
            [
                "run",
                "--detach",
                "--rm",
                "--name",
                manager,
                "--network",
                network,
                "--group-add",
                _docker_socket_group_id(),
                "-e",
                f"CODING_ASSISTANT_MANAGER_AUTH_SECRET={manager_auth_secret}",
                "-e",
                f"CODING_ASSISTANT_HOST_DATA_DIR={tmp_path}",
                "-e",
                f"CODING_ASSISTANT_WORKER_IMAGE={image}",
                "-e",
                f"CODING_ASSISTANT_WORKER_NETWORK={network}",
                "-e",
                f"OPENAI_BASE_URL=http://{fake_openai}:8000/v1",
                "-e",
                "OPENAI_API_KEY=test-key",
                "-v",
                "/var/run/docker.sock:/var/run/docker.sock",
                "-v",
                f"{tmp_path}:/data",
                "-p",
                "127.0.0.1::8764",
                image,
            ],
            timeout=30,
        )
        assert manager_started.returncode == 0, manager_started.stderr or manager_started.stdout
        endpoint = f"ws://127.0.0.1:{_published_port(manager, 8764)}"
        try:
            await _wait_for_manager_endpoint(endpoint, auth_token=manager_auth_secret)
            async with _connect_initialized(endpoint, auth_token=manager_auth_secret) as websocket:
                first_session_id = await _new_session(websocket, request_id=2)
                second_session_id = await _new_session(websocket, request_id=3)
                await websocket.send(
                    jsonrpc_request(4, "session/set_model", _scope_params(first_session_id, model="fake-model"))
                )
                first_set_model = await _receive_response(websocket, request_id=4)
                await websocket.send(
                    jsonrpc_request(5, "session/set_model", _scope_params(second_session_id, model="fake-model"))
                )
                second_set_model = await _receive_response(websocket, request_id=5)
                (tmp_path / "sessions" / first_session_id / "workspace" / "smoke.txt").write_text(
                    "alpha from first workspace",
                    encoding="utf-8",
                )
                (tmp_path / "sessions" / second_session_id / "workspace" / "smoke.txt").write_text(
                    "bravo from second workspace",
                    encoding="utf-8",
                )

                first_prompt = await _prompt_session(websocket, request_id=6, session_id=first_session_id)
                second_prompt = await _prompt_session(websocket, request_id=7, session_id=second_session_id)
                first_texts = await _load_session(websocket, request_id=8, session_id=first_session_id)
                second_texts = await _load_session(websocket, request_id=9, session_id=second_session_id)

            assert first_set_model["result"]["_meta"]["model"] == "fake-model"
            assert second_set_model["result"]["_meta"]["model"] == "fake-model"
            assert first_prompt["result"] == {"stopReason": "end_turn"}
            assert second_prompt["result"] == {"stopReason": "end_turn"}
            assert "tool result: alpha from first workspace" in first_texts
            assert "tool result: bravo from second workspace" in second_texts
        except Exception:
            worker_logs = _docker(["logs", worker_container], timeout=20)
            manager_logs = _docker(["logs", manager], timeout=20)
            fake_logs = _docker(["logs", fake_openai], timeout=20)
            pytest.fail(
                "\n".join(
                    [
                        "Manager end-to-end smoke failed.",
                        "manager logs:",
                        manager_logs.stdout or manager_logs.stderr,
                        "worker logs:",
                        worker_logs.stdout or worker_logs.stderr,
                        "fake OpenAI logs:",
                        fake_logs.stdout or fake_logs.stderr,
                    ],
                ),
            )
    finally:
        _remove_container(manager)
        _remove_container(worker_container)
        _remove_container(fake_openai)
        _remove_network(network)
        _remove_image(image)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_docker_worker_runner_reports_real_container_start_failure(tmp_path: Path) -> None:
    _require_docker()
    suffix = uuid.uuid4().hex[:12]
    session_id = f"fail-{suffix}"
    container_name = f"coding-assistant-worker-{session_id}"
    config = DockerWorkerConfig(
        image=f"missing-coding-assistant-image:{suffix}",
        network="bridge",
        startup_timeout=0.1,
    )
    runner = DockerWorkerRunner(config=config)
    prompt = WorkerPrompt(
        session_id=session_id,
        history=[],
        model="test-model",
        workspace=str(tmp_path),
        attachments=str(tmp_path / "attachments"),
        prompt=[],
    )

    try:
        with pytest.raises(DockerWorkerError, match="Failed to start worker container"):
            await runner.run_prompt(prompt=prompt, on_update=_ignore_update)
        inspected = _docker(["container", "inspect", container_name])
        assert inspected.returncode != 0
    finally:
        _remove_container(container_name)


@pytest.mark.asyncio
async def test_run_command_reports_missing_docker_command() -> None:
    with pytest.raises(DockerWorkerError, match="Docker command not found"):
        await _run_command(["definitely-not-a-real-docker-command"])


async def _ignore_update(update: SessionUpdate) -> None:
    del update
