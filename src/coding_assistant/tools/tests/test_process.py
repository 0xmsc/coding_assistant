import asyncio
import os
import shlex
from pathlib import Path

import pytest

from coding_assistant.tools.process import start_process


@pytest.mark.asyncio
async def test_start_process_env_merging() -> None:
    # Set a unique env var in the parent process
    os.environ["PARENT_VAR"] = "parent_value"

    # Define a new var to be merged
    extra_env = {"EXTRA_VAR": "extra_value"}

    # Run a command that prints both environment variables
    # We use python -c for cross-platform compatibility if needed,
    # but here we know we are in a unix-like environment.
    cmd = ["python3", "-c", "import os; print(os.environ.get('PARENT_VAR')); print(os.environ.get('EXTRA_VAR'))"]

    handle = await start_process(args=cmd, env=extra_env)
    await handle.wait(timeout=5.0)

    output = handle.stdout.strip().split("\n")

    assert "parent_value" in output
    assert "extra_value" in output


@pytest.mark.asyncio
async def test_start_process_env_override() -> None:
    os.environ["OVERRIDE_VAR"] = "original"

    # Override the existing var
    extra_env = {"OVERRIDE_VAR": "new_value"}

    cmd = ["python3", "-c", "import os; print(os.environ.get('OVERRIDE_VAR'))"]

    handle = await start_process(args=cmd, env=extra_env)
    await handle.wait(timeout=5.0)

    assert handle.stdout.strip() == "new_value"


@pytest.mark.asyncio
async def test_start_process_no_env_provided() -> None:
    os.environ["STAY_VAR"] = "stay"

    cmd = ["python3", "-c", "import os; print(os.environ.get('STAY_VAR'))"]

    # Pass None as env
    handle = await start_process(args=cmd, env=None)
    await handle.wait(timeout=5.0)

    assert handle.stdout.strip() == "stay"


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
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError(f"Child process {child_pid} is still running after terminate().")
