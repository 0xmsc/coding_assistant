from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal


MAX_RETAINED_OUTPUT_BYTES = 5_000_000


class OutputBuffer:
    """Continuously retain a bounded amount of subprocess output."""

    def __init__(self, stream: asyncio.StreamReader, *, max_bytes: int = MAX_RETAINED_OUTPUT_BYTES):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        self._stream = stream
        self._max_bytes = max_bytes
        self._buffer = bytearray()
        self._dropped_bytes = 0
        self._read_task = asyncio.create_task(self._read_stream())

    async def _read_stream(self) -> None:
        """Drain the stream until EOF."""
        while True:
            chunk = await self._stream.read(4096)
            if not chunk:
                break
            self._buffer.extend(chunk)
            if len(self._buffer) > self._max_bytes:
                discarded = len(self._buffer) - self._max_bytes // 2
                del self._buffer[:discarded]
                self._dropped_bytes += discarded

    @staticmethod
    def _format_output(*, content: bytes, dropped_bytes: int) -> str:
        parts: list[str] = []
        if dropped_bytes:
            parts.append(f"[output limit reached: {dropped_bytes} earlier bytes discarded]")
        if content:
            parts.append(content.decode(errors="replace"))
        return "\n".join(parts)

    @property
    def text(self) -> str:
        """Return all retained output as decoded text."""
        return self._format_output(content=bytes(self._buffer), dropped_bytes=self._dropped_bytes)

    def consume_text(self) -> str:
        """Return retained output and clear the buffer."""
        content = bytes(self._buffer)
        self._buffer.clear()
        dropped_bytes = self._dropped_bytes
        self._dropped_bytes = 0
        return self._format_output(content=content, dropped_bytes=dropped_bytes)

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

        if os.name == "posix":
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
        else:
            self._process.terminate()
        await self.wait(timeout=5.0)
        if not self.is_running:
            return

        if os.name == "posix":
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
        else:
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

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=stdin,
        env=env,
        start_new_session=os.name == "posix",
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


@dataclass(frozen=True)
class ManagedProcessResult:
    """Result of executing a subprocess tracked by the background task manager."""

    task_id: int
    status: Literal["background", "timeout", "finished"]
    output: str = ""
    exit_code: int | None = None
    truncated: bool = False


async def execute_managed_process(
    *,
    args: Sequence[str],
    task_name: str,
    register_task: Callable[[str, ProcessHandle], int],
    timeout: int,
    truncate_at: int,
    background: bool,
    stdin_input: str | None = None,
    env: dict[str, str] | None = None,
) -> ManagedProcessResult:
    """Execute and track a subprocess with shared timeout and cancellation behavior."""
    handle = await start_process(args=args, stdin_input=stdin_input, env=env)
    task_id = register_task(task_name, handle)

    if background:
        return ManagedProcessResult(task_id=task_id, status="background")

    try:
        finished = await handle.wait(timeout=timeout)
    except asyncio.CancelledError:
        await handle.terminate()
        raise
    if not finished:
        return ManagedProcessResult(task_id=task_id, status="timeout")

    output = handle.stdout
    return ManagedProcessResult(
        task_id=task_id,
        status="finished",
        output=truncate_output(output, truncate_at),
        exit_code=handle.exit_code,
        truncated=len(output) > truncate_at,
    )


def truncate_output(result: str, truncate_at: int) -> str:
    """Trim long output and append a note that records the original length."""
    if len(result) > truncate_at:
        note = f"\n\n[truncated output at: {truncate_at}, full length: {len(result)}]"
        truncated = result[: max(0, truncate_at - len(note))]
        return truncated + note

    return result
