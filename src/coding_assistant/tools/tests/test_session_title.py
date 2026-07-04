from __future__ import annotations

import pytest

from coding_assistant.llm.types import TextToolResult
from coding_assistant.tools.session_title import SessionTitleState, SetSessionTitleTool


@pytest.mark.asyncio
async def test_set_session_title_updates_finish_metadata() -> None:
    state = SessionTitleState()
    tool = SetSessionTitleTool(state=state)

    result = await tool.execute({"title": "  Debug upload flow  "})

    assert result == TextToolResult(content="Session title set to: Debug upload flow")
    assert state.finish_metadata() == {"title": "Debug upload flow"}


@pytest.mark.asyncio
async def test_set_session_title_rejects_empty_title() -> None:
    state = SessionTitleState()
    tool = SetSessionTitleTool(state=state)

    with pytest.raises(ValueError, match="Title must not be empty."):
        await tool.execute({"title": "   "})

    assert state.finish_metadata() is None
