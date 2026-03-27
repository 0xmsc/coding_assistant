from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence


class OutputBuffer:
    """Continuously read subprocess output into an in-memory buffer."""

    def __init__(self, stream: asyncio.StreamReader):
        self._stream = stream
        self._buffer = bytearray()
        self._read_task = asyncio.create_task(self._read_stream())

    async def _read_stream(self) -> None:
        """Drain the stream until EOF."""
        while True:
            chunk = await self._stream.read(4096)
            if not chunk:
                break
            self._buffer.extend(chunk)

    @property
    def text(self) -> str:
        """Return all buffered output as decoded text."""
        return self._buffer.decode(errors="replace")

    def consume_text(self) -> str:
        """Return buffered output and clear the buffer."""
        content = self._buffer.decode(errors="replace")
        self._buffer.clear()
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
        *,
        process: asyncio.subprocess.Process,
        output: OutputBuffer,
    ) -> None:
        self._process = process
        self._output = output

    @property
    def exit_code(self) -> int | None:
        """Return the process exit code, or `None` while it is still running."""
        return self._process.returncode

    @property
    def stdout(self) -> str:
        """Return all output captured so far."""
        return self._output.text

    @property
    def is_running(self) -> bool:
        """Return whether the process is still running."""
        return self.exit_code is None

    def consume_text(self) -> str:
        """Return and clear the output accumulated since the last read."""
        return self._output.consume_text()

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait for process exit and return `False` on timeout."""
        try:
            await asyncio.wait_for(self._process.wait(), timeout=timeout)
            await self._output.wait_for_finish()
            return True
        except asyncio.TimeoutError:
            return False

    async def terminate(self) -> None:
        """Try graceful termination first, then kill if needed."""
        if not self.is_running:
            return

        self._process.terminate()
        await self.wait(timeout=5.0)
        if not self.is_running:
            return

        self._process.kill()
        await self.wait(timeout=5.0)


async def start_process(
    *,
    args: Sequence[str],
    stdin_input: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcessHandle:
    """Start a process and return a handle to it."""
    stdin = asyncio.subprocess.PIPE if stdin_input is not None else asyncio.subprocess.DEVNULL

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=stdin,
        env=merged_env,
    )

    assert process.stdout is not None
    output = OutputBuffer(process.stdout)

    if stdin_input is not None:
        assert process.stdin is not None
        process.stdin.write(stdin_input.encode())
        await process.stdin.drain()
        process.stdin.close()
        await process.stdin.wait_closed()

    return ProcessHandle(process=process, output=output)


def truncate_output(result: str, truncate_at: int) -> str:
    """Trim long output and append a note that records the original length."""
    if len(result) > truncate_at:
        note = f"\n\n[truncated output at: {truncate_at}, full length: {len(result)}]"
        truncated = result[: max(0, truncate_at - len(note))]
        return truncated + note

    return result
