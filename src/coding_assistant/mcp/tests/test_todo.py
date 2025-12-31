from typing import Any
import pytest
from coding_assistant.mcp.todo import TodoManager


@pytest.fixture()
def manager() -> Any:
    return TodoManager()


def test_add_and_list_single(manager: TodoManager) -> None:
    r1 = manager.add(["Write tests"])
    assert r1.strip() == "- [ ] 1: Write tests"
    r2 = manager.add(["Refactor code"])
    lines = r2.splitlines()
    assert "- [ ] 1: Write tests" in lines
    assert any(line.endswith("2: Refactor code") for line in lines)

    text = manager.list_todos()
    assert "1: Write tests" in text
    assert "2: Refactor code" in text


def test_complete(manager: TodoManager) -> None:
    manager.add(["Implement feature"])
    manager.add(["Write docs"])
    complete_res = manager.complete(1)
    assert complete_res.startswith("- [x] 1: Implement feature")
    assert complete_res.count("Implement feature") == 1
    assert "- [ ] 2: Write docs" in complete_res
    text = manager.list_todos()
    assert "- [x] 1: Implement feature" in text
    assert "- [ ] 2: Write docs" in text


def test_complete_with_result(manager: TodoManager) -> None:
    manager.add(["Run benchmarks"])
    manager.add(["Prepare release notes"])
    manager.complete(1, result="Throughput +12% vs baseline")

    listing = manager.list_todos()
    assert "- [x] 1: Run benchmarks -> Throughput +12% vs baseline" in listing
    assert "- [ ] 2: Prepare release notes" in listing


def test_complete_invalid(manager: TodoManager) -> None:
    assert manager.complete(1) == "TODO 1 not found."
    manager.add(["Something"])
    assert manager.complete(99) == "TODO 99 not found."


def test_add_multiple_and_invalid(manager: TodoManager) -> None:
    out = manager.add(["A", "B"])
    lines = out.splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("1: A")
    assert lines[1].endswith("2: B")

    with pytest.raises(ValueError):
        manager.add([""])


def test_complete_ignores_empty_result(manager: TodoManager) -> None:
    manager.add(["Do something"])
    res = manager.complete(1, result="")  # empty result should be ignored
    assert res.startswith("- [x] 1: Do something")
    listing = manager.list_todos()
    assert "- [x] 1: Do something ->" not in listing
    assert "- [x] 1: Do something" in listing
