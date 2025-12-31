import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.llm.types import UserMessage

@pytest.mark.asyncio
async def test_run_chat_loop_exits_on_keyboard_interrupt_at_prompt():
    """Test that run_chat_loop terminates when KeyboardInterrupt is raised during ui.prompt."""
    history = [UserMessage(content="start")]
    model = "test-model"
    tools = []
    instructions = None
    
    # Mock UI to raise KeyboardInterrupt when prompt is called
    ui = AsyncMock()
    ui.prompt.side_effect = KeyboardInterrupt()
    
    # Run chat loop
    # If the fix works, this should return without raising KeyboardInterrupt
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
    # If we are here, it means the loop broke as expected
