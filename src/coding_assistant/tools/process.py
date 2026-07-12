from __future__ import annotations

import asyncio
import os
import signal
from collections import deque
from collections.abc import Sequence


DEFAULT_PROCESS_ENV_KEYS = {
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "PATH",
    "SHELL",
    "TERM",
    "USER",
}

MAX_RETAINED_OUTPUT_BYTES = 5_000_000


class OutputBuffer:
    """Continuously retain a bounded amount of subprocess output."""

    def __init__(self, stream: asyncio.StreamReader, *, max_bytes: int = MAX_RETAINED_OUTPUT_BYTES):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive.")
        self._stream = stream
        self._max_bytes = max_bytes
        self._chunks: deque[bytes] = deque()
        self._buffered_bytes = 0
        self._dropped_bytes = 0
        self._read_task = asyncio.create_task(self._read_stream())

    async def _read_stream(self) -> None:
        """Drain the stream until EOF."""
        while True:
            chunk = await self._stream.read(4096)
            if not chunk:
                break
            self._chunks.append(chunk)
            self._buffered_bytes += len(chunk)
            self._discard_excess()

    def _discard_excess(self) -> None:
        """Discard the oldest bytes once the retention limit is exceeded."""
        overflow = self._buffered_bytes - self._max_bytes
        while overflow > 0:
            chunk = self._chunks[0]
            discarded = min(len(chunk), overflow)
            if discarded == len(chunk):
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[discarded:]
            self._buffered_bytes -= discarded
            self._dropped_bytes += discarded
            overflow -= discarded

    @staticmethod
    def _format_output(*, content: bytes, dropped_bytes: int, remaining_bytes: int = 0) -> str:
        parts: list[str] = []
        if dropped_bytes:
            parts.append(f"[output limit reached: {dropped_bytes} earlier bytes discarded]")
        if content:
            parts.append(content.decode(errors="replace"))
        if remaining_bytes:
            parts.append(f"[{remaining_bytes} unread output bytes remain; call tasks_get_output again]")
        return "\n".join(parts)

    @property
    def text(self) -> str:
        """Return all retained output as decoded text."""
        return self._format_output(content=b"".join(self._chunks), dropped_bytes=self._dropped_bytes)

    def consume_text(self, max_bytes: int) -> str:
        """Return one output page, leaving later bytes available for the next read."""
        remaining = min(max_bytes, self._buffered_bytes)
        content = bytearray()
        while remaining > 0:
            chunk = self._chunks[0]
            consumed = min(len(chunk), remaining)
            content.extend(chunk[:consumed])
            if consumed == len(chunk):
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[consumed:]
            self._buffered_bytes -= consumed
            remaining -= consumed

        dropped_bytes = self._dropped_bytes
        self._dropped_bytes = 0
        return self._format_output(
            content=bytes(content),
            dropped_bytes=dropped_bytes,
            remaining_bytes=self._buffered_bytes,
        )

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

    def consume_text(self, max_bytes: int) -> str:
        """Return one page of output accumulated since the last read."""
        return self._output.consume_text(max_bytes)

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


def default_process_env() -> dict[str, str]:
    """Return the default scrubbed environment for tool subprocesses."""
    return {key: value for key, value in os.environ.items() if key in DEFAULT_PROCESS_ENV_KEYS}


async def start_process(
    *,
    args: Sequence[str],
    stdin_input: str | None = None,
    env: dict[str, str] | None = None,
) -> ProcessHandle:
    """Start a process and return a handle to it."""
    stdin = asyncio.subprocess.PIPE if stdin_input is not None else asyncio.subprocess.DEVNULL

    merged_env = default_process_env()
    if env:
        merged_env.update(env)

    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=stdin,
        env=merged_env,
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


def truncate_output(result: str, truncate_at: int) -> str:
    """Trim long output and append a note that records the original length."""
    if len(result) > truncate_at:
        note = f"\n\n[truncated output at: {truncate_at}, full length: {len(result)}]"
        truncated = result[: max(0, truncate_at - len(note))]
        return truncated + note

    return result
