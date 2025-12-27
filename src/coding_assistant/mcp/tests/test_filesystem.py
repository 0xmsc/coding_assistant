from pathlib import Path

import pytest

from coding_assistant.mcp.filesystem import edit_file, write_file


@pytest.mark.asyncio
async def test_write_file_creates_and_writes(tmp_path: Path):
    p = tmp_path / "a.txt"
    msg = await write_file(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    assert "Successfully wrote file" in msg and "a.txt" in msg


@pytest.mark.asyncio
async def test_write_file_overwrites_existing(tmp_path: Path):
    p = tmp_path / "b.txt"
    await write_file(p, "first")
    await write_file(p, "second")
    assert p.read_text(encoding="utf-8") == "second"


@pytest.mark.asyncio
async def test_write_file_creates_parent_directories(tmp_path: Path):
    p = tmp_path / "nested/dir/c.txt"
    assert not p.parent.exists()
    await write_file(p, "content")
    assert p.exists()
    assert p.read_text(encoding="utf-8") == "content"


@pytest.mark.asyncio
async def test_write_file_utf8_content(tmp_path: Path):
    p = tmp_path / "utf8.txt"
    text = "こんにちは世界 🌍"
    await write_file(p, text)
    assert p.read_text(encoding="utf-8") == text


@pytest.mark.asyncio
async def test_edit_file_unique_replace_and_diff(tmp_path: Path):
    p = tmp_path / "sample.txt"
    original = "hello world\nsecond line\n"
    await write_file(p, original)

    diff = await edit_file(p, old_text="world", new_text="Earth")

    assert p.read_text(encoding="utf-8") == "hello Earth\nsecond line\n"

    assert "@@" in diff
    assert "-hello world" in diff
    assert "+hello Earth" in diff


@pytest.mark.asyncio
async def test_edit_file_no_match_raises(tmp_path: Path):
    p = tmp_path / "nomatch.txt"
    await write_file(p, "abc\n")

    with pytest.raises(ValueError) as ei:
        await edit_file(p, old_text="zzz", new_text="yyy")
    assert "not found" in str(ei.value)


@pytest.mark.asyncio
async def test_edit_file_multiple_matches_raises(tmp_path: Path):
    p = tmp_path / "multi.txt"
    await write_file(p, "foo bar foo\n")

    with pytest.raises(ValueError) as ei:
        await edit_file(p, old_text="foo", new_text="baz")
    assert "multiple times" in str(ei.value)


@pytest.mark.asyncio
async def test_edit_file_multiple_edits_success(tmp_path: Path):
    p = tmp_path / "multi_success.txt"
    original = "alpha beta gamma\n"
    await write_file(p, original)

    diff1 = await edit_file(p, old_text="beta", new_text="BETA")
    diff2 = await edit_file(p, old_text="gamma", new_text="GAMMA")

    assert p.read_text(encoding="utf-8") == "alpha BETA GAMMA\n"
    assert "@@" in diff1 and "-alpha beta gamma" in diff1 and "+alpha BETA gamma" in diff1
    assert "@@" in diff2 and "-alpha BETA gamma" in diff2 and "+alpha BETA GAMMA" in diff2


@pytest.mark.asyncio
async def test_edit_file_order_applies_sequentially(tmp_path: Path):
    p = tmp_path / "order.txt"
    await write_file(p, "foo bar\n")

    await edit_file(p, old_text="foo", new_text="baz")
    diff2 = await edit_file(p, old_text="baz", new_text="FOO")

    assert p.read_text(encoding="utf-8") == "FOO bar\n"
    assert "+FOO bar" in diff2


@pytest.mark.asyncio
async def test_edit_file_atomicity_on_failure(tmp_path: Path):
    p = tmp_path / "atomic.txt"
    original = "one two three two\n"
    await write_file(p, original)

    await edit_file(p, old_text="one", new_text="ONE")

    with pytest.raises(ValueError) as ei:
        await edit_file(p, old_text="two", new_text="TWO")
    assert "multiple times" in str(ei.value)

    assert p.read_text(encoding="utf-8") == "ONE two three two\n"


@pytest.mark.asyncio
async def test_edit_file_empty_string_replacement(tmp_path: Path):
    """Test replacing with empty string as a form of deletion."""
    p = tmp_path / "empty_noop.txt"
    original = "content\n"
    await write_file(p, original)

    diff = await edit_file(p, old_text=original, new_text="")

    assert p.read_text(encoding="utf-8") == ""
    assert "-content" in diff


@pytest.mark.asyncio
async def test_edit_file_replace_with_empty_string(tmp_path: Path):
    p = tmp_path / "delete.txt"
    original = "keep delete keep\n"
    await write_file(p, original)

    diff = await edit_file(p, old_text=" delete", new_text="")

    assert p.read_text(encoding="utf-8") == "keep delete keep\n".replace(" delete", "")
    assert "-keep delete keep" in diff and "+keep keep" in diff


@pytest.mark.asyncio
async def test_edit_file_unicode_replacement(tmp_path: Path):
    p = tmp_path / "unicode.txt"
    original = "こんにちは世界\n"
    await write_file(p, original)

    diff = await edit_file(p, old_text="世界", new_text="World 🌍")

    assert p.read_text(encoding="utf-8") == "こんにちはWorld 🌍\n"
    assert "-こんにちは世界" in diff and "+こんにちはWorld 🌍" in diff


@pytest.mark.asyncio
async def test_edit_file_replace_entire_content(tmp_path: Path):
    p = tmp_path / "entire.txt"
    original = "entire content\n"
    await write_file(p, original)

    diff = await edit_file(p, old_text=original, new_text="")

    assert p.read_text(encoding="utf-8") == ""
    assert f"-{original.strip()}" in diff and "+" not in diff.splitlines()[-1]
