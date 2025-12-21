from __future__ import annotations

import asyncio
from typing import Annotated

from fastmcp import FastMCP

from coding_assistant_mcp.utils import truncate_output

shell_server = FastMCP()


async def _wait_for_process(proc, timeout) -> bool:
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


async def execute(
    command: Annotated[str, "The shell command to execute. Do not include 'bash -c'."],
    timeout: Annotated[int, "The timeout for the command in seconds."] = 30,
    truncate_at: Annotated[int, "Maximum number of characters to return in stdout/stderr combined."] = 50_000,
) -> str:
    """Execute a shell command using bash and return combined stdout/stderr."""
    command = command.strip()

    try:
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
        )

        assert proc.stdout is not None

        stdout_buf = bytearray()
        read_task = asyncio.create_task(_read_stream(stdout_buf, proc.stdout))
        timed_out = await _wait_for_process(proc, timeout=timeout)

        try:
            await asyncio.wait_for(read_task, timeout=5)
        except asyncio.TimeoutError:
            pass

        stdout = bytes(stdout_buf)
        stdout_text = stdout.decode(errors="replace")
        stdout_text = truncate_output(stdout_text, truncate_at)

        if timed_out:
            result = f"Command timed out after {timeout} seconds.\n\n{stdout_text}"
        elif proc.returncode != 0:
            result = f"Returncode: {proc.returncode}.\n\n{stdout_text}"
        else:
            result = stdout_text

    except Exception as e:
        result = f"Error: {str(e)}"

    return result


shell_server.tool(execute)
