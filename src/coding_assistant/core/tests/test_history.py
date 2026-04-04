import pytest

from coding_assistant.core.history import compact_history
from coding_assistant.llm.types import SystemMessage, UserMessage


def test_compact_history_preserves_system_prompt() -> None:
    history = [
        SystemMessage(content="# Instructions\n\nTest instructions"),
        UserMessage(content="Original task"),
    ]

    compacted = compact_history(history, "Short summary")

    assert [message.role for message in compacted] == ["system", "user"]
    assert isinstance(compacted[1].content, str)
    assert "Short summary" in compacted[1].content


def test_compact_history_requires_system_message() -> None:
    with pytest.raises(RuntimeError, match="initial system message"):
        compact_history([UserMessage(content="Hi")], "Short summary")
