from __future__ import annotations

import asyncio
from typing import Sequence
import os


class OutputBuffer:
    """Continuously read subprocess output into an in-memory buffer."""

    def __init__(self, stream: asyncio.StreamReader):
        self._stream = stream
        self._buf = bytearray()
        self._read_task = asyncio.create_task(self._read_stream())

    async def _read_stream(self) -> None:
        """Drain the stream until EOF."""
        while True:
            chunk = await self._stream.read(4096)
            if not chunk:
                break
            self._buf.extend(chunk)

    @property
    def text(self) -> str:
        """Return all buffered output as decoded text."""
        return self._buf.decode(errors="replace")

    def consume_text(self) -> str:
        """Return buffered output and clear the buffer."""
        content = self._buf.decode(errors="replace")
        self._buf.clear()
        return content

    async def wait_for_finish(self, timeout: float | None = 5.0) -> None:
        """Wait briefly for the background reader to finish draining output."""
        try:
            await asyncio.wait_for(self._read_task, timeout=timeout)
        except asyncio.TimeoutError:
            pass


class ProcessHandle:
    """Live handle for a subprocess and its captured combined output."""

    def __init__(
        self,
        proc: asyncio.subprocess.Process,
        output: OutputBuffer,
    ):
        self.proc = proc
        self.output = output

    @property
    def exit_code(self) -> int | None:
        """Return the process exit code, or `None` while it is still running."""
        return self.proc.returncode

    @property
    def stdout(self) -> str:
        """Return all output captured so far."""
        return self.output.text

    @property
    def is_running(self) -> bool:
        """Return whether the process is still running."""
        return self.exit_code is None

    def consume_text(self) -> str:
        """Return and clear the output accumulated since the last read."""
        return self.output.consume_text()

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait for process exit and return `False` on timeout."""
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=timeout)
            await self.output.wait_for_finish()
            return True
        except asyncio.TimeoutError:
            return False

    async def terminate(self) -> None:
        """Try graceful termination first, then kill if needed."""
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
