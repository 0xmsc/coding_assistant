from pathlib import Path
from unittest.mock import patch

import pytest

from coding_assistant.history_manager import HistoryManager
from coding_assistant.llm.types import UserMessage


@pytest.mark.asyncio
async def test_history_manager_saves() -> None:
    manager = HistoryManager(context_name="test")

    with patch("coding_assistant.history_manager.save_orchestrator_history") as mock_save:
        manager.start()
        await manager.save_orchestrator_history(
            working_directory=Path("/tmp"),
            history=[UserMessage(content="hi")],
        )
        await manager.stop()

    mock_save.assert_called_once()
