from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class ProcessResult:
    stdout: str
    returncode: int | None
    timed_out: bool


class ProcessHandle:
    """
    A handle to a running or completed process that allows shared access
    to its output buffer and state.
    """

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        stdout_buf: bytearray,
        read_task: asyncio.Task,
    ):
        self.proc = proc
        self.stdout_buf = stdout_buf
        self.read_task = read_task
        self._timed_out = False

    @property
    def returncode(self) -> int | None:
        return self.proc.returncode

    @property
    def stdout(self) -> str:
        return self.stdout_buf.decode(errors="replace")

    @property
    def is_running(self) -> bool:
        return self.returncode is None

    async def wait(self, timeout: float | None = None) -> bool:
        """
        Wait for process to finish.
        Returns True if process finished, False if timed out.
        """
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=timeout)
            # Give the read task a moment to finish flushing
            try:
                await asyncio.wait_for(self.read_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            return True
        except asyncio.TimeoutError:
            return False

    async def terminate(self):
        if self.is_running:
            self.proc.terminate()
            await self.wait(timeout=5.0)
            if self.is_running:
                self.proc.kill()
                await self.wait()


async def _read_stream(buf: bytearray, stream: asyncio.StreamReader):
    while True:
        try:
            chunk = await stream.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
            # Keep buffer size under control (100KB limit for background tasks)
            if len(buf) > 100_000:
                del buf[: len(buf) - 100_000]
        except Exception:
            break


async def start_process(
    args: Sequence[str],
    stdin_input: str | None = None,
) -> ProcessHandle:
    """Start a process and return a handle to it."""

    stdin = asyncio.subprocess.PIPE if stdin_input is not None else asyncio.subprocess.DEVNULL

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=stdin,
    )

    stdout_buf = bytearray()
    assert proc.stdout is not None
    read_task = asyncio.create_task(_read_stream(stdout_buf, proc.stdout))

    if stdin_input is not None:
        assert proc.stdin is not None
        try:
            proc.stdin.write(stdin_input.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            await proc.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError):
            pass

    return ProcessHandle(proc, stdout_buf, read_task)


async def execute_process(
    args: Sequence[str],
    stdin_input: str | None = None,
    timeout: int = 30,
) -> ProcessResult:
    """Execute a process and return its output and status (original API)."""
    handle = await start_process(args, stdin_input=stdin_input)
    finished = await handle.wait(timeout=timeout)

    if not finished:
        await handle.terminate()
        return ProcessResult(
            stdout=handle.stdout,
            returncode=handle.returncode,
            timed_out=True,
        )

    return ProcessResult(
        stdout=handle.stdout,
        returncode=handle.returncode,
        timed_out=False,
    )
