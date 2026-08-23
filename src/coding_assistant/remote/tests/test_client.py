from __future__ import annotations

import pytest

from coding_assistant.remote.client import WorkerClient, _finish_title


def test_finish_title_extracts_and_strips_title() -> None:
    assert _finish_title({"title": "  My Title  "}) == "My Title"
    assert _finish_title({"title": "   "}) is None
    assert _finish_title({"title": 123}) is None
    assert _finish_title(None) is None
    assert _finish_title({}) is None


@pytest.mark.asyncio
async def test_worker_client_run_validates_session_id() -> None:
    client = object.__new__(WorkerClient)
    with pytest.raises(ValueError, match="sessionId"):
        await client.run({})
