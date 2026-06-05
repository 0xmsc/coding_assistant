from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from time import monotonic

from coding_assistant.core.session_updates import SessionUpdate
from coding_assistant.manager.remote_worker import RemoteWorkerRunner
from coding_assistant.manager.service import WorkerCommit, WorkerPrompt
from coding_assistant.remote.client import RemoteClientEvent, RemoteSessionClient


class DockerWorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class DockerCommandResult:
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class DockerWorkerConfig:
    image: str
    model: str
    network: str
    container_prefix: str = "coding-assistant-worker-"
    worker_port: int = 8765
    workspace_mount: str = "/workspace"
    docker_command: str = "docker"
    startup_timeout: float = 15.0
    environment: dict[str, str] = field(default_factory=dict)
    instructions: tuple[str, ...] = ()
    skills_directories: tuple[str, ...] = ()


class DockerWorkerRunner:
    """Start one Docker worker container for each active manager prompt."""

    def __init__(self, *, config: DockerWorkerConfig) -> None:
        self._config = config
        self._active_runners: dict[str, RemoteWorkerRunner] = {}
        self._active_containers: dict[str, str] = {}
        self._active_lock = asyncio.Lock()

    async def run_prompt(
        self,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerCommit:
        container_name = _container_name(prefix=self._config.container_prefix, session_id=prompt.session_id)
        endpoint = f"ws://{container_name}:{self._config.worker_port}"
        runner = RemoteWorkerRunner(endpoint=endpoint)
        await self._remove_container(container_name, allow_failure=True)
        await self._start_container(container_name=container_name, workspace=prompt.workspace)
        await self._register_active(session_id=prompt.session_id, container_name=container_name, runner=runner)
        try:
            await self._wait_until_ready(endpoint)
            worker_prompt = replace(prompt, workspace=self._config.workspace_mount)
            return await runner.run_prompt(prompt=worker_prompt, on_update=on_update)
        finally:
            await self._unregister_active(session_id=prompt.session_id, runner=runner)
            await self._remove_container(container_name, allow_failure=True)

    async def cancel(self, *, session_id: str) -> None:
        async with self._active_lock:
            runner = self._active_runners.get(session_id)
        if runner is None:
            return
        await runner.cancel(session_id=session_id)

    async def _register_active(
        self,
        *,
        session_id: str,
        container_name: str,
        runner: RemoteWorkerRunner,
    ) -> None:
        async with self._active_lock:
            self._active_runners[session_id] = runner
            self._active_containers[session_id] = container_name

    async def _unregister_active(self, *, session_id: str, runner: RemoteWorkerRunner) -> None:
        async with self._active_lock:
            if self._active_runners.get(session_id) is runner:
                self._active_runners.pop(session_id, None)
                self._active_containers.pop(session_id, None)

    async def _start_container(self, *, container_name: str, workspace: str) -> None:
        args = _docker_run_args(config=self._config, container_name=container_name, workspace=workspace)
        result = await _run_command(args)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise DockerWorkerError(f"Failed to start worker container {container_name}: {detail}")

    async def _remove_container(self, container_name: str, *, allow_failure: bool) -> None:
        result = await _run_command([self._config.docker_command, "rm", "-f", container_name])
        if result.returncode == 0 or allow_failure:
            return
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise DockerWorkerError(f"Failed to remove worker container {container_name}: {detail}")

    async def _wait_until_ready(self, endpoint: str) -> None:
        deadline = monotonic() + self._config.startup_timeout
        last_error = "worker did not respond"
        while monotonic() < deadline:
            try:
                client = await RemoteSessionClient.connect(
                    endpoint=endpoint,
                    on_event=_ignore_remote_event,
                    on_disconnect=_ignore_disconnect,
                )
                try:
                    await client.initialize()
                finally:
                    await client.close()
                return
            except Exception as exc:
                last_error = str(exc)
                await asyncio.sleep(0.1)
        raise DockerWorkerError(f"Worker endpoint {endpoint} did not become ready: {last_error}")


def _container_name(*, prefix: str, session_id: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", session_id).strip("-")
    return f"{prefix}{normalized or 'session'}"


def _docker_run_args(*, config: DockerWorkerConfig, container_name: str, workspace: str) -> list[str]:
    args = [
        config.docker_command,
        "run",
        "--detach",
        "--rm",
        "--name",
        container_name,
        "--network",
        config.network,
        "--workdir",
        config.workspace_mount,
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{workspace}:{config.workspace_mount}:rw",
        "-e",
        f"PORT={config.worker_port}",
    ]
    for key, value in config.environment.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend(
        [
            config.image,
            "coding-assistant-worker",
            "--model",
            config.model,
            "--host",
            "0.0.0.0",
            "--port",
            str(config.worker_port),
            "--workspace",
            config.workspace_mount,
        ],
    )
    if config.instructions:
        args.extend(["--instructions", *config.instructions])
    if config.skills_directories:
        args.extend(["--skills-directories", *config.skills_directories])
    return args


async def _ignore_remote_event(event: RemoteClientEvent) -> None:
    del event


async def _ignore_disconnect(endpoint: str) -> None:
    del endpoint


async def _run_command(args: Sequence[str]) -> DockerCommandResult:
    env = os.environ.copy()
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError as exc:
        raise DockerWorkerError(f"Docker command not found: {args[0]}") from exc
    stdout_bytes, stderr_bytes = await process.communicate()
    return DockerCommandResult(
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        returncode=process.returncode or 0,
    )
