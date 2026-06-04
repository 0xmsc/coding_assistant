from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.llm.types import AssistantMessage
from coding_assistant.manager import docker_worker
from coding_assistant.manager.docker_worker import (
    DockerCommandResult,
    DockerWorkerConfig,
    DockerWorkerError,
    DockerWorkerRunner,
    _docker_run_args,
    _run_command,
)
from coding_assistant.manager.service import WorkerCommit, WorkerPrompt


class FakeRemoteWorkerRunner:
    instances: list[FakeRemoteWorkerRunner] = []

    def __init__(self, *, endpoint: str) -> None:
        self.endpoint = endpoint
        self.prompts: list[WorkerPrompt] = []
        self.cancelled_session_ids: list[str] = []
        FakeRemoteWorkerRunner.instances.append(self)

    async def run_prompt(
        self,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerCommit:
        del on_update
        self.prompts.append(prompt)
        return WorkerCommit(messages=[AssistantMessage(content="done")], stop_reason="end_turn")

    async def cancel(self, *, session_id: str) -> None:
        self.cancelled_session_ids.append(session_id)


def test_docker_run_args_mount_session_workspace_and_start_worker() -> None:
    config = DockerWorkerConfig(
        image="coding-assistant:test",
        model="test-model",
        network="assistant-net",
        environment={"OPENAI_API_KEY": "test-key"},
        instructions=("Be concise.",),
        skills_directories=("/skills",),
        mcp_servers=('{"name":"test","command":"server"}',),
    )

    args = _docker_run_args(config=config, container_name="coding-assistant-worker-sess", workspace="/data/ws/sess")

    assert args[:4] == ["docker", "run", "--detach", "--rm"]
    assert ["--name", "coding-assistant-worker-sess"] == args[4:6]
    assert "--cap-drop" in args
    assert "ALL" in args
    assert "--security-opt" in args
    assert "no-new-privileges" in args
    assert "-v" in args
    assert "/data/ws/sess:/workspace:rw" in args
    assert "coding-assistant:test" in args
    assert "coding-assistant-worker" in args
    assert args[args.index("--network") + 1] == "assistant-net"
    assert args[args.index("--model") + 1] == "test-model"
    assert args[args.index("--host") + 1] == "0.0.0.0"
    assert args[args.index("--workspace") + 1] == "/workspace"
    assert "OPENAI_API_KEY=test-key" in args
    assert "Be concise." in args
    assert "/skills" in args


@pytest.mark.asyncio
async def test_docker_worker_runner_starts_delegates_and_removes_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    FakeRemoteWorkerRunner.instances = []

    async def fake_run_command(args: list[str]) -> DockerCommandResult:
        commands.append(args)
        return DockerCommandResult(stdout="", stderr="", returncode=0)

    async def fake_wait_until_ready(self: DockerWorkerRunner, endpoint: str) -> None:
        del self, endpoint

    monkeypatch.setattr(docker_worker, "_run_command", fake_run_command)
    monkeypatch.setattr(docker_worker, "RemoteWorkerRunner", FakeRemoteWorkerRunner)
    monkeypatch.setattr(DockerWorkerRunner, "_wait_until_ready", fake_wait_until_ready)

    runner = DockerWorkerRunner(
        config=DockerWorkerConfig(image="coding-assistant:test", model="test-model", network="assistant-net"),
    )
    prompt = WorkerPrompt(
        session_id="sess_test",
        base_version=0,
        history=[],
        workspace=str(tmp_path),
        prompt=[],
    )

    result = await runner.run_prompt(prompt=prompt, on_update=_ignore_update)

    assert result.stop_reason == "end_turn"
    assert [command[:3] for command in commands] == [
        ["docker", "rm", "-f"],
        ["docker", "run", "--detach"],
        ["docker", "rm", "-f"],
    ]
    assert FakeRemoteWorkerRunner.instances[0].endpoint == "ws://coding-assistant-worker-sess_test:8765"
    assert FakeRemoteWorkerRunner.instances[0].prompts == [prompt]


@pytest.mark.asyncio
async def test_docker_worker_runner_cancel_forwards_to_active_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRemoteWorkerRunner.instances = []

    async def fake_run_command(args: list[str]) -> DockerCommandResult:
        del args
        return DockerCommandResult(stdout="", stderr="", returncode=0)

    async def fake_wait_until_ready(self: DockerWorkerRunner, endpoint: str) -> None:
        del self, endpoint

    async def blocking_run_prompt(
        self: FakeRemoteWorkerRunner,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerCommit:
        del prompt, on_update
        await runner.cancel(session_id="sess_test")
        return WorkerCommit(messages=[], stop_reason="cancelled")

    monkeypatch.setattr(docker_worker, "_run_command", fake_run_command)
    monkeypatch.setattr(docker_worker, "RemoteWorkerRunner", FakeRemoteWorkerRunner)
    monkeypatch.setattr(DockerWorkerRunner, "_wait_until_ready", fake_wait_until_ready)
    monkeypatch.setattr(FakeRemoteWorkerRunner, "run_prompt", blocking_run_prompt)

    runner = DockerWorkerRunner(
        config=DockerWorkerConfig(image="coding-assistant:test", model="test-model", network="assistant-net"),
    )
    prompt = WorkerPrompt(session_id="sess_test", base_version=0, history=[], workspace="/tmp/ws", prompt=[])

    result = await runner.run_prompt(prompt=prompt, on_update=_ignore_update)

    assert result.stop_reason == "cancelled"
    assert FakeRemoteWorkerRunner.instances[0].cancelled_session_ids == ["sess_test"]


@pytest.mark.asyncio
async def test_docker_worker_runner_reports_container_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_command(args: list[str]) -> DockerCommandResult:
        if args[:3] == ["docker", "run", "--detach"]:
            return DockerCommandResult(stdout="", stderr="docker daemon unavailable", returncode=1)
        return DockerCommandResult(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(docker_worker, "_run_command", fake_run_command)

    runner = DockerWorkerRunner(
        config=DockerWorkerConfig(image="coding-assistant:test", model="test-model", network="assistant-net"),
    )
    prompt = WorkerPrompt(session_id="sess_test", base_version=0, history=[], workspace="/tmp/ws", prompt=[])

    with pytest.raises(DockerWorkerError, match="docker daemon unavailable"):
        await runner.run_prompt(prompt=prompt, on_update=_ignore_update)


@pytest.mark.asyncio
async def test_run_command_reports_missing_docker_command() -> None:
    with pytest.raises(DockerWorkerError, match="Docker command not found"):
        await _run_command(["definitely-not-a-real-docker-command"])


async def _ignore_update(update: SessionUpdate) -> None:
    del update
