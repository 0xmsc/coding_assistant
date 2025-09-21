import pytest

from coding_assistant_mcp.shell import execute, set_shell_confirmation_patterns


@pytest.mark.asyncio
async def test_shell_execute_timeout():
    out = await execute(command="sleep 2", timeout=1)
    assert "timed out" in out


@pytest.mark.asyncio
async def test_shell_execute_nonzero_return_code():
    out = await execute(command="bash -lc 'exit 7'")
    assert out.startswith("Returncode: 7\n\n")


@pytest.mark.asyncio
async def test_shell_execute_truncates_output():
    out = await execute(command="yes 1 | head -c 1000", truncate_at=200)
    assert out.endswith("\n\n[truncated output due to truncate_at limit]")
    assert len(out) <= 200


@pytest.mark.asyncio
async def test_shell_execute_happy_path_stdout():
    out = await execute(command="printf 'hello'", timeout=5)
    assert out == "hello"


@pytest.mark.asyncio
async def test_shell_execute_stderr_captured_with_zero_exit():
    # Writes to stderr, then exits 0
    out = await execute(command="echo 'oops' >&2; true", timeout=5)
    assert out == "oops\n"


@pytest.mark.asyncio
async def test_shell_execute_nonzero_with_stderr_content():
    out = await execute(command="echo 'bad' >&2; exit 4", timeout=5)
    assert out.startswith("Returncode: 4\n\n")
    assert "bad\n" in out


@pytest.mark.asyncio
async def test_shell_execute_requires_confirmation_yes(monkeypatch):
    set_shell_confirmation_patterns([r"^echo "])  # matches

    async def yes(_command: str) -> bool:
        return True

    monkeypatch.setattr("coding_assistant_mcp.shell._ask_confirmation", yes)
    out = await execute(command="echo hello")
    assert out == "hello\n"


@pytest.mark.asyncio
async def test_shell_execute_requires_confirmation_no(monkeypatch):
    set_shell_confirmation_patterns([r"^echo "])  # matches

    async def no(_command: str) -> bool:
        return False

    monkeypatch.setattr("coding_assistant_mcp.shell._ask_confirmation", no)
    out = await execute(command="echo hello")
    assert out == "Command execution denied."


@pytest.mark.asyncio
async def test_shell_execute_no_match_no_confirmation_needed():
    set_shell_confirmation_patterns([r"^echo foo"])
    out = await execute(command="echo bar")
    assert out == "bar\n"
