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


@pytest.mark.asyncio
async def test_stop_mcp_server_timeout_protection():
    """Test that _stop_mcp_server doesn't hang forever if a task is stubborn."""
    from coding_assistant.main import _stop_mcp_server

    async def stubborn_task():
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Simulate a task that takes a long time to clean up or "hangs"
            await asyncio.sleep(10)

    task = asyncio.create_task(stubborn_task())

    # This should return after ~2 seconds because of the timeout in _stop_mcp_server
    # without raising TimeoutError to the caller.
    with patch("coding_assistant.main.asyncio.timeout", side_effect=lambda t: asyncio.timeout(0.1)):
        # We patch the timeout to be very short for the test
        await _stop_mcp_server(task)

    # The task should have been told to cancel
    assert task.cancelling() > 0 or task.cancelled()
