import asyncio
import os
import shlex
from pathlib import Path

import pytest

from coding_assistant.tools.process import start_process


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


@pytest.mark.asyncio
async def test_start_process_uses_scrubbed_environment_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARENT_VAR", "parent_value")

    cmd = ["python3", "-c", "import os; print(os.environ.get('PARENT_VAR')); print(bool(os.environ.get('PATH')))"]

    handle = await start_process(args=cmd)
    await handle.wait(timeout=5.0)

    output = handle.stdout.strip().split("\n")

    assert output == ["None", "True"]


@pytest.mark.asyncio
async def test_start_process_explicit_env_is_added(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OVERRIDE_VAR", "original")
    extra_env = {"OVERRIDE_VAR": "new_value"}

    cmd = ["python3", "-c", "import os; print(os.environ.get('OVERRIDE_VAR'))"]

    handle = await start_process(args=cmd, env=extra_env)
    await handle.wait(timeout=5.0)

    assert handle.stdout.strip() == "new_value"


@pytest.mark.asyncio
async def test_start_process_does_not_expose_provider_or_manager_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CODING_ASSISTANT_MANAGER_AUTH_SECRET", "manager-secret")

    cmd = [
        "python3",
        "-c",
        (
            "import os; "
            "print(os.environ.get('OPENAI_API_KEY')); "
            "print(os.environ.get('OPENAI_BASE_URL')); "
            "print(os.environ.get('CODING_ASSISTANT_MANAGER_AUTH_SECRET'))"
        ),
    ]

    handle = await start_process(args=cmd)
    await handle.wait(timeout=5.0)

    assert handle.stdout.strip().split("\n") == ["None", "None", "None"]


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="process group termination is only exercised on POSIX")
async def test_terminate_kills_child_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    command = f"sleep 30 & echo $! > {shlex.quote(str(child_pid_path))}; wait"

    handle = await start_process(args=["bash", "-c", command])

    child_pid = None
    for _ in range(20):
        if child_pid_path.exists():
            raw_pid = child_pid_path.read_text().strip()
            if raw_pid:
                child_pid = int(raw_pid)
                break
        await asyncio.sleep(0.05)

    assert child_pid is not None

    await handle.terminate()
    assert await handle.wait(timeout=1.0) is True

    for _ in range(20):
        if not _process_is_running(child_pid):
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError(f"Child process {child_pid} is still running after terminate().")
