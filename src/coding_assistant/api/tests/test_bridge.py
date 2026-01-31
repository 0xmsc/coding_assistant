import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from coding_assistant.api.bridge import WebSocketUI
from coding_assistant.api.models import AnswerResponse, ConfirmationResponse

@pytest.mark.asyncio
async def test_websocket_ui_ask():
    # Setup
    mock_ws = AsyncMock()
    response_queue = asyncio.Queue()
    ui = WebSocketUI(mock_ws, response_queue)
    
    # Simulate a response appearing in the queue
    # We need to know the request_id generated inside ask()
    # For testing, we can mock uuid.uuid4 to return a fixed value
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("uuid.uuid4", lambda: "test-uuid")
        
        # Prepare the response
        resp = AnswerResponse(request_id="test-uuid", text="Hello World")
        await response_queue.put(resp)
        
        # Action
        result = await ui.ask("What is your name?")
        
        # Assertions
        assert result == "Hello World"
        mock_ws.send_text.assert_called_once()
        # Verify the sent JSON contains our payload
        sent_json = mock_ws.send_text.call_args[0][0]
        assert "What is your name?" in sent_json
        assert "test-uuid" in sent_json

@pytest.mark.asyncio
async def test_websocket_ui_confirm():
    # Setup
    mock_ws = AsyncMock()
    response_queue = asyncio.Queue()
    ui = WebSocketUI(mock_ws, response_queue)
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("uuid.uuid4", lambda: "test-uuid-confirm")
        
        # Prepare the response
        resp = ConfirmationResponse(request_id="test-uuid-confirm", value=True)
        await response_queue.put(resp)
        
        # Action
        result = await ui.confirm("Are you sure?")
        
        # Assertions
        assert result is True
        mock_ws.send_text.assert_called_once()
        assert "Are you sure?" in mock_ws.send_text.call_args[0][0]
