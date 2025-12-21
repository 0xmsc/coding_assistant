from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Sequence


@dataclass
class ProcessResult:
    stdout: str
    returncode: int | None
    timed_out: bool


async def _wait_for_process(proc: asyncio.subprocess.Process, timeout: int) -> bool:
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
        return False
    except asyncio.TimeoutError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        return True


async def _read_stream(buf: bytearray, stream: asyncio.StreamReader):
    while True:
        try:
            chunk = await stream.read(4096)
            if not chunk:
                break
            buf.extend(chunk)
        except Exception:
            break


async def execute_process(
    args: Sequence[str],
    stdin_input: str | None = None,
    timeout: int = 30,
) -> ProcessResult:
    """Execute a process and return its output and status."""

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
        proc.stdin.write(stdin_input.encode())
        await proc.stdin.drain()
        proc.stdin.close()
        await proc.stdin.wait_closed()

    timed_out = await _wait_for_process(proc, timeout=timeout)

    try:
        await asyncio.wait_for(read_task, timeout=5)
    except asyncio.TimeoutError:
        pass

    stdout = stdout_buf.decode(errors="replace")

    return ProcessResult(
        stdout=stdout,
        returncode=proc.returncode,
        timed_out=timed_out,
    )
