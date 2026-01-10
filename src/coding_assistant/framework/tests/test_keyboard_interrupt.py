from unittest.mock import AsyncMock

import pytest

from coding_assistant.framework.callbacks import NullToolCallbacks
from coding_assistant.llm.types import NullProgressCallbacks
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.llm.types import BaseMessage, Tool, UserMessage


@pytest.mark.asyncio
async def test_run_chat_loop_raises_keyboard_interrupt_at_prompt() -> None:
    """Test that run_chat_loop propagates KeyboardInterrupt raised during ui.prompt."""
    history: list[BaseMessage] = [UserMessage(content="start")]
    model = "test-model"
    tools: list[Tool] = []
    instructions = None

    # Mock UI to raise KeyboardInterrupt when prompt is called
    ui = AsyncMock()
    ui.prompt.side_effect = KeyboardInterrupt()

    # Verify that KeyboardInterrupt is raised
    with pytest.raises(KeyboardInterrupt):
        await run_chat_loop(
            history=history,
            model=model,
            tools=tools,
            instructions=instructions,
            callbacks=NullProgressCallbacks(),
            tool_callbacks=NullToolCallbacks(),
            completer=AsyncMock(),
            ui=ui,
            context_name="test",
        )

    # Verify prompt was called
    ui.prompt.assert_called_once()
