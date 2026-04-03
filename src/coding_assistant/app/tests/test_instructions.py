from pathlib import Path

from coding_assistant.app.instructions import get_instructions


def _get_project_root() -> Path:
    return Path(__file__).parent.parent.parent.parent.resolve()


def test_get_instructions_base_and_user_instructions(tmp_path: Path) -> None:
    wd = tmp_path
    instr = get_instructions(working_directory=wd, user_instructions=["  A  ", "B\n"])

    assert "Do not install any software" in instr
    assert "\nA\n" in instr
    # Second item may be at end without trailing newline
    assert "\nB\n" in instr or instr.rstrip().endswith("\nB") or instr.endswith("B")


def test_get_instructions_with_plan_and_local_file(tmp_path: Path) -> None:
    wd = tmp_path
    local_dir = wd / ".coding_assistant"
    local_dir.mkdir()
    (local_dir / "instructions.md").write_text("LOCAL OVERRIDE\n- extra rule")

    instr = get_instructions(working_directory=wd, user_instructions=[])

    assert "LOCAL OVERRIDE" in instr
    assert "- extra rule" in instr


def test_get_instructions_appends_extra_sections(tmp_path: Path) -> None:
    instr = get_instructions(
        working_directory=tmp_path,
        user_instructions=[],
        extra_sections=["# Local tools\n\n## Shell\n- Use shell_execute."],
    )

    assert "# Local tools" in instr
    assert "Use shell_execute." in instr
