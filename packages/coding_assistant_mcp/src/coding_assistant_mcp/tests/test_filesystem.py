from pathlib import Path

import pytest

from coding_assistant_mcp.filesystem import FilesystemManager


@pytest.mark.asyncio
async def test_write_file_creates_directories_and_diff(tmp_path: Path):
    base = tmp_path / "nested/dir/structure"
    p = base / "f.txt"

    m = FilesystemManager()
    msg = m.write_file(path=p, content="hello\n")
    assert p.exists()
    assert p.read_text(encoding="utf-8") == "hello\n"
    # unified diff from empty to one line
    assert msg.startswith(f"--- {p}\n+++")
    assert "\n+hello" in msg


@pytest.mark.asyncio
async def test_edit_file_basic_before_after_replace(tmp_path: Path):
    p = tmp_path / "f.txt"
    p.write_text("A\nB\nC\n", encoding="utf-8")

    m = FilesystemManager()

    # BEFORE: insert X before B
    msg1 = m.edit_file(path=p, pattern=r"^B$", text="X", position="before")
    assert "Edited with diff" in msg1
    assert p.read_text(encoding="utf-8") == "A\nX\nB\nC\n"

    # AFTER: insert Y after B
    m.edit_file(path=p, pattern=r"^B$", text="Y", position="after")
    assert p.read_text(encoding="utf-8") == "A\nX\nB\nY\nC\n"

    # REPLACE: replace Y with Z
    m.edit_file(path=p, pattern=r"^Y$", text="Z", position="replace")
    assert p.read_text(encoding="utf-8") == "A\nX\nB\nZ\nC\n"


@pytest.mark.asyncio
async def test_edit_file_enforce_unique_and_first_match(tmp_path: Path):
    p = tmp_path / "g.txt"
    p.write_text("A\nB\nA\nC\n", encoding="utf-8")

    m = FilesystemManager()
    # enforce uniqueness -> error
    with pytest.raises(ValueError):
        m.edit_file(path=p, pattern=r"^A$", text="X", position="before", enforce_unique_match=True)

    # allow non-unique -> choose first occurrence
    m.edit_file(path=p, pattern=r"^A$", text="X", position="after", enforce_unique_match=False)
    assert p.read_text(encoding="utf-8") == "A\nX\nB\nA\nC\n"


@pytest.mark.asyncio
async def test_undo_last_edit_messages(tmp_path: Path):
    p = tmp_path / "h.txt"
    p.write_text("a\nhello\nzzz\n", encoding="utf-8")

    m = FilesystemManager()
    # Without prior edit
    assert m.undo_last_edit() == "Nothing to undo."

    # Make an edit and then undo successfully
    m.edit_file(path=p, pattern=r"^hello$", text="WORLD", position="replace")
    before_undo = p.read_text(encoding="utf-8")
    diff = m.undo_last_edit()
    assert diff.startswith(f"--- {p}\n+++")
    assert p.read_text(encoding="utf-8") != before_undo


@pytest.mark.asyncio
async def test_undo_mismatch_error(tmp_path: Path):
    p = tmp_path / "j.txt"
    p.write_text("abc\nMID\nxyz\n", encoding="utf-8")

    m = FilesystemManager()
    m.copy_range(path=p, pattern=r"^MID$")
    m.paste(path=p, pattern=r"^xyz$", position="before")

    # Mutate file externally to break undo
    p.write_text(p.read_text(encoding="utf-8") + "!", encoding="utf-8")

    with pytest.raises(ValueError):
        m.undo_last_edit()


@pytest.mark.asyncio
async def test_linewise_multi_line_match_copy_and_cut(tmp_path: Path):
    p = tmp_path / "k.txt"
    p.write_text("A\nB\nC\nD\n", encoding="utf-8")
    m = FilesystemManager()

    # Match two lines via regex spanning a newline
    msg = m.copy_range(path=p, pattern=r"B\nC")
    assert msg.startswith("Copied\n\n")
    assert m.show_clipboard() == "B\nC"

    # Cut the same region
    msg2 = m.cut_range(path=p, pattern=r"B\nC")
    assert msg2.startswith("Cut\n\n") and "with diff" in msg2
    assert p.read_text(encoding="utf-8") == "A\nD\n"


@pytest.mark.asyncio
async def test_path_accepts_path_type(tmp_path: Path):
    p = tmp_path / "m.txt"
    m = FilesystemManager()

    # str path
    m.write_file(path=p, content="one\n")
    # Path path on edit
    m.edit_file(path=p, pattern=r"^one$", text="two", position="replace")
    assert p.read_text(encoding="utf-8") == "two\n"


@pytest.mark.asyncio
async def test_copy_and_cut_and_paste_and_undo_roundtrip(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text(
        """
start
alpha

beta
end
zzz
""".lstrip(),
        encoding="utf-8",
    )

    m = FilesystemManager()

    # copy alpha..beta by matching the whole lines via a multiline regex
    msg = m.copy_range(
        path=p,
        pattern=r"^alpha$\n\n^beta$",
        enforce_unique_match=True,
    )
    assert msg.startswith("Copied\n\n")
    clip = m.show_clipboard()
    assert clip.startswith("alpha")

    # cut the same region now
    msg2 = m.cut_range(
        path=p,
        pattern=r"^alpha$\n\n^beta$",
    )
    assert msg2.startswith("Cut\n\n")
    after_cut = p.read_text(encoding="utf-8")
    assert "alpha" not in after_cut

    # paste clipboard before zzz
    msg3 = m.paste(
        path=p,
        pattern=r"^zzz$",
        position="before",
    )
    assert msg3.startswith("Pasted\n\n")
    after_paste = p.read_text(encoding="utf-8")
    assert "alpha" in after_paste

    # undo last paste
    msg4 = m.undo_last_edit()
    assert msg4.startswith(f"--- {p}\n+++")
    after_undo = p.read_text(encoding="utf-8")
    assert "alpha" not in after_undo


@pytest.mark.asyncio
async def test_replace_multi_line_region(tmp_path):
    p = tmp_path / "g.txt"
    p.write_text(
        """
<begin>
A

B
<end>
C
""".lstrip(),
        encoding="utf-8",
    )

    m = FilesystemManager()
    # Copy a region (A blank B)
    m.copy_range(
        path=p,
        pattern=r"^A$\n\n^B$",
    )

    # Replace the begin line with clipboard (linewise replace of <begin> only)
    msg = m.paste(
        path=p,
        pattern=r"^<begin>$",
        position="after",
    )
    assert msg.startswith("Pasted\n\n")
    # Undo should bring back the previous region
    undo_msg = m.undo_last_edit()
    assert undo_msg.startswith(f"--- {p}\n+++")


@pytest.mark.asyncio
async def test_uniqueness_violation_and_first_match(tmp_path):
    p = tmp_path / "h.txt"
    p.write_text(
        """
X
Y
X
Y
Z
""".lstrip(),
        encoding="utf-8",
    )

    m = FilesystemManager()

    # Enforce unique start: should fail because ^X$ matches twice
    with pytest.raises(ValueError):
        m.copy_range(
            path=p,
            pattern=r"^X$",
            enforce_unique_match=True,
        )

    # Allow and implicitly pick the first occurrence
    msg = m.copy_range(
        path=p,
        pattern=r"^X$",
        enforce_unique_match=False,
    )
    assert msg.startswith("Copied\n\n")


@pytest.mark.asyncio
async def test_empty_clipboard_paste_error(tmp_path):
    p = tmp_path / "i.txt"
    p.write_text("hello", encoding="utf-8")

    m = FilesystemManager()

    with pytest.raises(ValueError):
        m.paste(path=p, pattern=r"^hello$", position="before")


@pytest.mark.asyncio
async def test_clear_clipboard_and_show_empty_error(tmp_path):
    p = tmp_path / "n.txt"
    p.write_text("ONE\n", encoding="utf-8")
    m = FilesystemManager()

    # Copy, then clear
    m.copy_range(path=p, pattern=r"^ONE$")
    assert m.show_clipboard() == "ONE"
    msg = m.clear_clipboard()
    assert msg == "Cleared clipboard."
    with pytest.raises(ValueError):
        m.show_clipboard()


@pytest.mark.asyncio
async def test_paste_with_position_replace(tmp_path: Path):
    p = tmp_path / "r.txt"
    p.write_text("A\nX\nB\n", encoding="utf-8")

    m = FilesystemManager()

    # Copy single line and replace a different single line
    m.copy_range(path=p, pattern=r"^X$")
    msg = m.paste(path=p, pattern=r"^A$", position="replace")
    assert msg.startswith("Pasted\n\n") and "with diff" in msg
    # After replacement, first line A is replaced by X; original X remains
    assert p.read_text(encoding="utf-8") == "X\nX\nB\n"

    # Uniqueness enforcement on paste 'replace' should error when pattern matches multiple times
    with pytest.raises(ValueError):
        m.paste(path=p, pattern=r"^X$", position="replace", enforce_unique_match=True)

    # Allow non-unique: first match is replaced
    m.copy_range(path=p, pattern=r"^B$")
    msg2 = m.paste(path=p, pattern=r"^X$", position="replace", enforce_unique_match=False)
    assert msg2.startswith("Pasted\n\n")
    assert p.read_text(encoding="utf-8") == "B\nX\nB\n"
