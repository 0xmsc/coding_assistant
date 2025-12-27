import os
import pytest
from coding_assistant.mcp.proc import start_process


@pytest.mark.asyncio
async def test_start_process_env_merging():
    # Set a unique env var in the parent process
    os.environ["PARENT_VAR"] = "parent_value"

    # Define a new var to be merged
    extra_env = {"EXTRA_VAR": "extra_value"}

    # Run a command that prints both environment variables
    # We use python -c for cross-platform compatibility if needed,
    # but here we know we are in a unix-like environment.
    cmd = ["python3", "-c", "import os; print(os.environ.get('PARENT_VAR')); print(os.environ.get('EXTRA_VAR'))"]

    handle = await start_process(cmd, env=extra_env)
    await handle.wait(timeout=5.0)

    output = handle.stdout.strip().split("\n")

    assert "parent_value" in output
    assert "extra_value" in output


@pytest.mark.asyncio
async def test_start_process_env_override():
    os.environ["OVERRIDE_VAR"] = "original"

    # Override the existing var
    extra_env = {"OVERRIDE_VAR": "new_value"}

    cmd = ["python3", "-c", "import os; print(os.environ.get('OVERRIDE_VAR'))"]

    handle = await start_process(cmd, env=extra_env)
    await handle.wait(timeout=5.0)

    assert handle.stdout.strip() == "new_value"


@pytest.mark.asyncio
async def test_start_process_no_env_provided():
    os.environ["STAY_VAR"] = "stay"

    cmd = ["python3", "-c", "import os; print(os.environ.get('STAY_VAR'))"]

    # Pass None as env
    handle = await start_process(cmd, env=None)
    await handle.wait(timeout=5.0)

    assert handle.stdout.strip() == "stay"
