import pytest
from coding_assistant_mcp.todo import reset_state, add, list_todos, complete


@pytest.fixture(autouse=True)
def todo_state_reset():
    reset_state()


def test_add_and_list_single():
    r1 = add(["Write tests"])
    assert r1.strip() == "- [ ] 1: Write tests"
    r2 = add(["Refactor code"])
    # After second add, output is the full list with both tasks
    lines = r2.splitlines()
    assert "- [ ] 1: Write tests" in lines
    assert any(l.endswith("2: Refactor code") for l in lines)

    text = list_todos()
    assert "1: Write tests" in text
    assert "2: Refactor code" in text


def test_complete():
    add(["Implement feature"])
    add(["Write docs"])
    complete_res = complete(1)
    assert complete_res.startswith("Completed TODO 1: Implement feature\n")
    assert "- [x] 1: Implement feature" in complete_res
    assert "- [ ] 2: Write docs" in complete_res
    # After completion, listing should show the completed task with an x (independently via list_todos)
    text = list_todos()
    assert "- [x] 1: Implement feature" in text
    assert "- [ ] 2: Write docs" in text


def test_complete_invalid():
    # No tasks yet
    res = complete(1)
    assert res.startswith("TODO 1 not found\n")
    add(["Something"])
    res2 = complete(99)
    assert res2.startswith("TODO 99 not found\n")


def test_add_multiple_and_invalid():
    out = add(["A", "B"])
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("1: A")
    assert lines[1].endswith("2: B")

    # Empty description should raise
    with pytest.raises(ValueError):
        add([""])
