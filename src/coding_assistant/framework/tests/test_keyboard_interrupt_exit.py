import asyncio
from asyncio import timeout as real_timeout
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from coding_assistant.framework.callbacks import NullProgressCallbacks, NullToolCallbacks
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.types import Tool
from coding_assistant.llm.types import BaseMessage, UserMessage
from coding_assistant.session import stop_mcp_server
from coding_assistant.tools.mcp_server import start_mcp_server


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


@pytest.mark.asyncio
async def test_mcp_server_shutdown_logic() -> None:
    """Test that start_mcp_server sets up the expected log configuration."""
    tools: list[Tool] = []
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
async def test_stop_mcp_server_timeout_protection() -> None:
    """Test that stop_mcp_server doesn't hang forever if a task is stubborn."""

    async def stubborn_task() -> None:
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Simulate a task that takes a long time to clean up or "hangs"
            await asyncio.sleep(10)

    task: asyncio.Task[Any] = asyncio.create_task(stubborn_task())

    # This should return because of the timeout in stop_mcp_server
    with patch("coding_assistant.session.asyncio.timeout", side_effect=lambda t: real_timeout(0.1)):
        await stop_mcp_server(task, NullProgressCallbacks())

    # The task should have been told to cancel
    assert task.cancelling() > 0 or task.cancelled()
