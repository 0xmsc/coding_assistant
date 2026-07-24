from __future__ import annotations

import pytest

from coding_assistant.remote.client import _run_result_from_jsonrpc


def test_run_result_rejects_unknown_stop_reason() -> None:
    with pytest.raises(RuntimeError, match="did not include a completed run result"):
        _run_result_from_jsonrpc(
            {"stopReason": "failed", "messages": []},
            method="session/prompt",
        )
