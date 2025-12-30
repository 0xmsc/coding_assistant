from pathlib import Path


from typing import cast

from coding_assistant.instructions import get_instructions
from coding_assistant.tools.mcp import MCPServer


def _get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.resolve()


def test_get_instructions_base_and_user_instructions(tmp_path: Path):
    wd = tmp_path
    instr = get_instructions(working_directory=wd, user_instructions=["  A  ", "B\n"])

    assert "Do not install any software" in instr
    assert "\nA\n" in instr
    # Second item may be at end without trailing newline
    assert "\nB\n" in instr or instr.rstrip().endswith("\nB") or instr.endswith("B")


def test_get_instructions_with_planning_and_local_file(tmp_path: Path):
    wd = tmp_path
    local_dir = wd / ".coding_assistant"
    local_dir.mkdir()
    (local_dir / "instructions.md").write_text("LOCAL OVERRIDE\n- extra rule")

    instr = get_instructions(working_directory=wd, user_instructions=[])

    assert "LOCAL OVERRIDE" in instr
    assert "- extra rule" in instr


def test_get_instructions_appends_mcp_instructions(tmp_path: Path):
    wd = tmp_path

    class _FakeServer:
        def __init__(self, name: str, instructions: str | None):
            self.name = name
            self.instructions = instructions

    s1 = _FakeServer("server1", "- Use server1 tools whenever possible.")
    s2 = _FakeServer("server2", "- Server2: prefer safe operations.")

    instr = get_instructions(
        working_directory=wd,
        user_instructions=[],
        mcp_servers=cast(list[MCPServer], [s1, s2]),
    )

    assert "Use server1 tools whenever possible." in instr
    assert "Server2: prefer safe operations." in instr


def test_get_instructions_ignores_empty_or_missing_mcp_instructions(tmp_path: Path):
    wd = tmp_path

    class _BlankServer:
        def __init__(self, name: str, instructions: str | None):
            self.name = name
            self.instructions = instructions

    s1 = _BlankServer("s1", "   ")  # only whitespace
    s2 = _BlankServer("s2", "")  # empty
    s3 = _BlankServer("s3", None)  # None

    instr = get_instructions(
        working_directory=wd,
        user_instructions=[],
        mcp_servers=cast(list[MCPServer], [s1, s2, s3]),
    )

    # Ensure baseline rule present and nothing from the servers leaked
    assert "Do not install any software" in instr
    assert "Server" not in instr


def test_get_instructions_includes_mcp_formatting_with_real_mcp_instructions(tmp_path: Path):
    wd = tmp_path
    mcp_instructions = """
## Shell
- Rule 1

## Tasks
- Rule 2
""".strip()

    class _FakeServer:
        def __init__(self, name: str, instructions: str | None):
            self.name = name
            self.instructions = instructions

    server = _FakeServer("coding_assistant.mcp", mcp_instructions)

    instr = get_instructions(
        working_directory=wd,
        user_instructions=[],
        mcp_servers=cast(list[MCPServer], [server]),
    )

    assert "# MCP `coding_assistant.mcp` instructions" in instr
    assert "## Shell" in instr
    assert "- Rule 1" in instr
    assert "## Tasks" in instr
    assert "- Rule 2" in instr
