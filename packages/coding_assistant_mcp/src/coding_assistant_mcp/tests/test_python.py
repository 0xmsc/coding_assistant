import pytest

from coding_assistant_mcp.python import execute


@pytest.mark.asyncio
async def test_python_run_timeout():
    out = await execute(code="import time; time.sleep(2)", timeout=1)
    assert "timed out" in out


@pytest.mark.asyncio
async def test_python_run_exception_includes_traceback():
    out = await execute(code="import sys; sys.exit(7)")
    assert out.startswith("Exception:\n\n")
    assert "SystemExit: 7" in out


@pytest.mark.asyncio
async def test_python_run_truncates_output():
    out = await execute(code="print('x'*1000)", truncate_at=200)
    assert "\n\n[truncated output at: " in out
    assert ", full length: " in out
    assert len(out) <= 200


@pytest.mark.asyncio
async def test_python_run_happy_path_stdout():
    out = await execute(code="print('hello', end='')", timeout=5)
    assert out == "hello"


@pytest.mark.asyncio
async def test_python_run_stderr_captured_with_zero_exit():
    out = await execute(code="import sys; sys.stderr.write('oops\\n')")
    assert out == "oops\n"


@pytest.mark.asyncio
async def test_python_run_exception_with_stderr_content():
    out = await execute(code="import sys; sys.stderr.write('bad\\n'); sys.exit(4)")
    assert out.startswith("Exception:\n\n")
    assert "bad\n" in out
