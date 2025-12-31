import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.llm.types import UserMessage
from coding_assistant.tools.mcp_server import start_mcp_server

@pytest.mark.asyncio
async def test_run_chat_loop_raises_keyboard_interrupt_at_prompt():
    """Test that run_chat_loop propagates KeyboardInterrupt raised during ui.prompt."""
    history = [UserMessage(content="start")]
    model = "test-model"
    tools = []
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

@pytest.mark.asyncio
async def test_mcp_server_shutdown_logic():
    """Test that start_mcp_server sets up the expected log configuration."""
    tools = []
    port = 9999
    
    with patch("fastmcp.FastMCP.run_async", new_callable=AsyncMock) as mock_run:
        task = await start_mcp_server(tools, port)
        
        # Verify run_async parameters
        _, kwargs = mock_run.call_args
        assert kwargs["log_level"] == "critical"
        assert "log_config" in kwargs["uvicorn_config"]
        assert kwargs["uvicorn_config"]["log_config"]["loggers"]["uvicorn.error"]["level"] == "CRITICAL"
        
        # Task is registered in the loop
        assert not task.done()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
