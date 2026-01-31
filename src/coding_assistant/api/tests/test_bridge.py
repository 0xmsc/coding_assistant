import asyncio
import json
import pytest
from typing import Any
from unittest.mock import AsyncMock
from coding_assistant.api.bridge import WebSocketUI
from coding_assistant.api.models import AnswerResponse, ConfirmationResponse


@pytest.mark.asyncio
async def test_websocket_ui_ask() -> None:
    # Setup
    mock_ws = AsyncMock()
    response_queue: asyncio.Queue[Any] = asyncio.Queue()
    ui = WebSocketUI(mock_ws, response_queue)

    # Start the ask call in the background
    ask_task = asyncio.create_task(ui.ask("What is your name?"))

    # Wait for the message to be sent
    # We poll briefly until mock_ws.send_text.called becomes true
    for _ in range(10):
        if mock_ws.send_text.called:
            break
        await asyncio.sleep(0.01)

    assert mock_ws.send_text.called
    sent_json = json.loads(mock_ws.send_text.call_args[0][0])
    request_id = sent_json["payload"]["request_id"]

    # Now provide the response with the MATCHING ID
    resp = AnswerResponse(request_id=request_id, text="Hello World")
    await response_queue.put(resp)

    # Await the result
    result = await ask_task

    # Assertions
    assert result == "Hello World"
    assert "What is your name?" in mock_ws.send_text.call_args[0][0]


@pytest.mark.asyncio
async def test_websocket_ui_confirm() -> None:
    # Setup
    mock_ws = AsyncMock()
    response_queue: asyncio.Queue[Any] = asyncio.Queue()
    ui = WebSocketUI(mock_ws, response_queue)

    # Start the confirm call in the background
    confirm_task = asyncio.create_task(ui.confirm("Are you sure?"))

    # Wait for the message to be sent
    for _ in range(10):
        if mock_ws.send_text.called:
            break
        await asyncio.sleep(0.01)

    assert mock_ws.send_text.called
    sent_json = json.loads(mock_ws.send_text.call_args[0][0])
    request_id = sent_json["payload"]["request_id"]

    # Now provide the response with the MATCHING ID
    resp = ConfirmationResponse(request_id=request_id, value=True)
    await response_queue.put(resp)

    # Await the result
    result = await confirm_task

    # Assertions
    assert result is True
    assert "Are you sure?" in mock_ws.send_text.call_args[0][0]
