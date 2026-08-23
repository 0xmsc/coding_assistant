import asyncio
import os
import shlex
from pathlib import Path

import pytest

from coding_assistant.tools.process import OutputBuffer, start_process


@pytest.mark.asyncio
async def test_output_buffer_bounds_retained_output() -> None:
    stream = asyncio.StreamReader()
    output = OutputBuffer(stream, max_bytes=10)
    stream.feed_data(b"0123456789abcdef")
    stream.feed_eof()
    await output.wait_for_finish()

    first = output.consume_text()
    second = output.consume_text()

    assert first == "[output limit reached: 11 earlier bytes discarded]\nbcdef"
    assert second == ""


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False

    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.exists():
        state = stat_path.read_text().split()[2]
        return state != "Z"

    return True


async def _read_pid_when_ready(path: Path) -> int:
    for _ in range(20):
        if path.exists():
            raw_pid = path.read_text().strip()
            if raw_pid:
                return int(raw_pid)
        await asyncio.sleep(0.05)
    raise AssertionError(f"PID was not written to {path}.")


async def _wait_until_not_running(pid: int) -> None:
    for _ in range(20):
        if not _process_is_running(pid):
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Child process {pid} is still running after terminate().")


@pytest.mark.asyncio
async def test_start_process_inherits_environment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARENT_VAR", "parent_value")

    cmd = ["python3", "-c", "import os; print(os.environ.get('PARENT_VAR')); print(bool(os.environ.get('PATH')))"]

    handle = await start_process(args=cmd)
    await handle.wait(timeout=5.0)

    output = handle.stdout.strip().split("\n")

    assert output == ["parent_value", "True"]


@pytest.mark.asyncio
async def test_start_process_uses_explicit_env_when_provided(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARENT_VAR", "parent_value")
    custom_env = {"CUSTOM_VAR": "custom_value"}

    cmd = ["python3", "-c", "import os; print(os.environ.get('PARENT_VAR')); print(os.environ.get('CUSTOM_VAR'))"]

    handle = await start_process(args=cmd, env=custom_env)
    await handle.wait(timeout=5.0)

    output = handle.stdout.strip().split("\n")

    assert output == ["None", "custom_value"]


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="process group termination is only exercised on POSIX")
async def test_terminate_kills_child_process_group(tmp_path: Path) -> None:
    shell_pid_path = tmp_path / "shell.pid"
    child_pid_path = tmp_path / "child.pid"
    shell_pid_file = shlex.quote(str(shell_pid_path))
    child_pid_file = shlex.quote(str(child_pid_path))
    command = "; ".join(
        [
            f"echo $$ > {shell_pid_file}",
            f"sleep 30 & echo $! > {child_pid_file}",
            "wait",
        ]
    )

    handle = await start_process(args=["bash", "-c", command])

    shell_pid = await _read_pid_when_ready(shell_pid_path)
    child_pid = await _read_pid_when_ready(child_pid_path)
    assert os.getpgid(child_pid) == os.getpgid(shell_pid)

    await handle.terminate()
    assert await handle.wait(timeout=1.0) is True
    await _wait_until_not_running(child_pid)
