from __future__ import annotations

import subprocess
import textwrap
import uuid
from pathlib import Path

import pytest

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.manager.docker_worker import (
    DockerWorkerConfig,
    DockerWorkerError,
    DockerWorkerRunner,
    _docker_run_args,
    _run_command,
)
from coding_assistant.manager.service import WorkerPrompt


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
@pytest.mark.slow
async def test_docker_worker_runner_starts_real_worker_container(tmp_path: Path) -> None:
    _require_docker()
    suffix = uuid.uuid4().hex[:12]
    image = f"coding-assistant:test-{suffix}"
    network = f"coding-assistant-test-{suffix}"
    fake_openai = f"coding-assistant-fake-openai-{suffix}"
    manager = f"coding-assistant-manager-smoke-{suffix}"
    session_id = f"sess-{suffix}"
    worker_container = f"coding-assistant-worker-{session_id}"

    try:
        build = _docker(["build", "-t", image, "."], timeout=180)
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

        script = textwrap.dedent(
            """
            import asyncio
            import os

            from coding_assistant.core.session_updates import SessionUpdate
            from coding_assistant.llm.types import SystemMessage
            from coding_assistant.manager.docker_worker import DockerWorkerConfig, DockerWorkerRunner
            from coding_assistant.manager.service import WorkerPrompt

            async def ignore_update(update: SessionUpdate) -> None:
                pass

            async def main() -> None:
                assert os.geteuid() != 0
                assert os.environ["SMOKE_IMAGE_USER"]
                runner = DockerWorkerRunner(
                    config=DockerWorkerConfig(
                        image=os.environ["SMOKE_IMAGE"],
                        model="fake-model",
                        network=os.environ["SMOKE_NETWORK"],
                        startup_timeout=30,
                        environment={
                            "OPENAI_BASE_URL": os.environ["SMOKE_OPENAI_BASE_URL"],
                            "OPENAI_API_KEY": "test-key",
                        },
                    ),
                )
                commit = await runner.run_prompt(
                    prompt=WorkerPrompt(
                        session_id=os.environ["SMOKE_SESSION_ID"],
                        base_version=0,
                        history=[SystemMessage(content="system")],
                        workspace=os.environ["SMOKE_WORKSPACE"],
                        prompt=[{"type": "text", "text": "docker smoke"}],
                    ),
                    on_update=ignore_update,
                )
                contents = [getattr(message, "content", None) for message in commit.messages]
                assert commit.stop_reason == "end_turn", commit
                assert contents == ["docker smoke", "fake response: docker smoke"], contents

            asyncio.run(main())
            """,
        )
        image_user = _docker(["image", "inspect", image, "--format", "{{.Config.User}}"], timeout=20)
        assert image_user.returncode == 0, image_user.stderr or image_user.stdout
        assert image_user.stdout.strip(), image_user.stdout
        manager_result = _docker(
            [
                "run",
                "--rm",
                "--name",
                manager,
                "--network",
                network,
                "--group-add",
                _docker_socket_group_id(),
                "-v",
                "/var/run/docker.sock:/var/run/docker.sock",
                "-v",
                f"{tmp_path}:{tmp_path}",
                "-e",
                f"SMOKE_IMAGE={image}",
                "-e",
                f"SMOKE_IMAGE_USER={image_user.stdout.strip()}",
                "-e",
                f"SMOKE_NETWORK={network}",
                "-e",
                f"SMOKE_OPENAI_BASE_URL=http://{fake_openai}:8000/v1",
                "-e",
                f"SMOKE_SESSION_ID={session_id}",
                "-e",
                f"SMOKE_WORKSPACE={tmp_path}",
                image,
                "python",
                "-c",
                script,
            ],
            timeout=120,
        )
        if manager_result.returncode != 0:
            worker_logs = _docker(["logs", worker_container], timeout=20)
            fake_logs = _docker(["logs", fake_openai], timeout=20)
            pytest.fail(
                "\n".join(
                    [
                        "Manager smoke container failed.",
                        "manager stdout:",
                        manager_result.stdout,
                        "manager stderr:",
                        manager_result.stderr,
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
        model="test-model",
        network="bridge",
        startup_timeout=0.1,
    )
    runner = DockerWorkerRunner(config=config)
    prompt = WorkerPrompt(session_id=session_id, base_version=0, history=[], workspace=str(tmp_path), prompt=[])

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
