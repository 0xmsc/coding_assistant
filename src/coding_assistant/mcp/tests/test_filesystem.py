from pathlib import Path

import pytest

from coding_assistant.mcp.filesystem import edit_file, write_file


@pytest.mark.asyncio
async def test_edit_file_replace_all_false_default(tmp_path: Path):
    """Test that replace_all=False (default) behaves as before, rejecting multiple matches."""
    p = tmp_path / "replace_all_false.txt"
    original = "foo bar foo\n"
    await write_file(p, original)

    with pytest.raises(ValueError) as ei:
        await edit_file(p, old_text="foo", new_text="baz", replace_all=False)
    assert "multiple times" in str(ei.value)

    # Ensure file unchanged
    assert p.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_edit_file_replace_all_true_multiple_occurrences(tmp_path: Path):
    """Test that replace_all=True replaces all occurrences of old_text."""
    p = tmp_path / "replace_all_true.txt"
    original = "foo bar foo baz foo\n"
    await write_file(p, original)

    diff = await edit_file(p, old_text="foo", new_text="XYZ", replace_all=True)

    expected = "XYZ bar XYZ baz XYZ\n"
    assert p.read_text(encoding="utf-8") == expected

    # Check diff reflects all changes
    assert "@@" in diff
    assert "-foo bar foo baz foo" in diff
    assert "+XYZ bar XYZ baz XYZ" in diff


@pytest.mark.asyncio
async def test_edit_file_replace_all_true_single_occurrence(tmp_path: Path):
    """Test that replace_all=True works correctly when there's only one occurrence."""
    p = tmp_path / "replace_all_single.txt"
    original = "hello unique world\n"
    await write_file(p, original)

    diff = await edit_file(p, old_text="unique", new_text="special", replace_all=True)

    expected = "hello special world\n"
    assert p.read_text(encoding="utf-8") == expected
    assert "-hello unique world" in diff
    assert "+hello special world" in diff


@pytest.mark.asyncio
async def test_edit_file_replace_all_true_no_occurrences(tmp_path: Path):
    """Test that replace_all=True with no matches still raises ValueError."""
    p = tmp_path / "replace_all_none.txt"
    original = "no match here\n"
    await write_file(p, original)

    with pytest.raises(ValueError) as ei:
        await edit_file(p, old_text="missing", new_text="replaced", replace_all=True)
    assert "not found" in str(ei.value)

    # Ensure unchanged
    assert p.read_text(encoding="utf-8") == original
