from __future__ import annotations

import asyncio
from typing import Sequence
import os


class OutputBuffer:
    def __init__(self, stream: asyncio.StreamReader):
        self._stream = stream
        self._buf = bytearray()
        self._read_task = asyncio.create_task(self._read_stream())

    async def _read_stream(self):
        while True:
            chunk = await self._stream.read(4096)
            if not chunk:
                break
            self._buf.extend(chunk)

    @property
    def text(self) -> str:
        return self._buf.decode(errors="replace")

    async def wait_for_finish(self, timeout: float | None = 5.0):
        try:
            await asyncio.wait_for(self._read_task, timeout=timeout)
        except asyncio.TimeoutError:
            pass


class ProcessHandle:
    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        output: OutputBuffer,
    ):
        self.proc = proc
        self.output = output

    @property
    def exit_code(self) -> int | None:
        return self.proc.returncode

    @property
    def stdout(self) -> str:
        return self.output.text

    @property
    def is_running(self) -> bool:
        return self.exit_code is None

    async def wait(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=timeout)
            await self.output.wait_for_finish()
            return True
        except asyncio.TimeoutError:
            return False

    async def terminate(self):
        if not self.is_running:
            return

        self.proc.terminate()
        await self.wait(timeout=5.0)
        if not self.is_running:
            return

        self.proc.kill()
        await self.wait(timeout=5.0)


async def start_process(
    args: Sequence[str],
    stdin_input: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcessHandle:
    """Start a process and return a handle to it."""

    stdin = asyncio.subprocess.PIPE if stdin_input is not None else asyncio.subprocess.DEVNULL

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=stdin,
        env=merged_env,
    )

    assert proc.stdout is not None
    output = OutputBuffer(proc.stdout)

    if stdin_input is not None:
        assert proc.stdin is not None
        proc.stdin.write(stdin_input.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()

    return ProcessHandle(proc, output)
