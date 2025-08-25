import os
from pathlib import Path

import pytest

from coding_assistant.instructions import get_instructions


def test_get_instructions_base_and_user_instructions(tmp_path: Path):
    wd = tmp_path
    # No local file, no planning
    instr = get_instructions(working_directory=wd, plan=False, user_instructions=["  A  ", "B\n"])

    # Key baseline rules should be present
    assert "Do not initialize a new git repository" in instr
    # User instructions appended, trimmed and in order
    assert "\nA\n" in instr
    # Second item may be at end without trailing newline
    assert "\nB\n" in instr or instr.rstrip().endswith("\nB") or instr.endswith("B")


def test_get_instructions_with_planning_and_local_file(tmp_path: Path):
    wd = tmp_path
    local_dir = wd / ".coding_assistant"
    local_dir.mkdir()
    (local_dir / "instructions.md").write_text("LOCAL OVERRIDE\n- extra rule")

    instr = get_instructions(working_directory=wd, plan=True, user_instructions=[])

    # Planning block should be included
    assert "You are in planning mode." in instr
    # Local instructions appended
    assert "LOCAL OVERRIDE" in instr
    assert "- extra rule" in instr
