import asyncio
import uuid
from typing import Optional

from fastapi import WebSocket

from coding_assistant.api.models import (
    AskRequest,
    ConfirmRequest,
    Envelope,
    StatusMessage,
    ContentChunk,
    ToolStart,
    ToolResult,
)
from coding_assistant.ui import UI
from coding_assistant.llm.types import ProgressCallbacks, StatusLevel


class WebSocketUI(UI):
    def __init__(self, websocket: WebSocket, response_queue: asyncio.Queue):
        self.websocket = websocket
        self.response_queue = response_queue

    async def _send_and_wait(self, request: AskRequest | ConfirmRequest) -> Any:
        # Send the request over the websocket
        envelope = Envelope(payload=request)
        await self.websocket.send_text(envelope.model_dump_json())

        # Wait for the client to send a response with the same request_id
        # The SessionManager or Server logic will feed this queue
        while True:
            response = await self.response_queue.get()
            if response.request_id == request.request_id:
                return response
            # If it's for a different request (unlikely in single-task flow), 
            # we might need to handle it or re-queue it. For now, assume sequential.

    async def ask(self, prompt_text: str, default: Optional[str] = None) -> str:
        request_id = str(uuid.uuid4())
        request = AskRequest(request_id=request_id, prompt=prompt_text, default=default)
        response = await self._send_and_wait(request)
        return response.text

    async def confirm(self, prompt_text: str) -> bool:
        request_id = str(uuid.uuid4())
        request = ConfirmRequest(request_id=request_id, prompt=prompt_text)
        response = await self._send_and_wait(request)
        return response.value

    async def prompt(self, words: list[str] | None = None) -> str:
        # For headless, 'prompt' usually doesn't make sense as a separate action 
        # from 'start', but we implement it as 'ask' for compatibility.
        return await self.ask("General prompt requested")


class WebSocketProgressCallbacks(ProgressCallbacks):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def _send(self, payload: Any):
        envelope = Envelope(payload=payload)
        await self.websocket.send_text(envelope.model_dump_json())

    def on_status_message(self, message: str, level: StatusLevel = StatusLevel.INFO) -> None:
        # ProgressCallbacks are currently synchronous in the framework (not awaited)
        # We need to bridge this to async. 
        # Option 1: Use asyncio.create_task 
        # Option 2: Update the framework to support async callbacks (cleaner but more changes)
        # For now, we utilize the event loop to send.
        payload = StatusMessage(level=level.value, message=message)
        asyncio.create_task(self._send(payload))

    def on_content_chunk(self, content: str) -> None:
        payload = ContentChunk(content=content)
        asyncio.create_task(self._send(payload))

    def on_tool_start(self, call_id: str, name: str, arguments: dict[str, Any]) -> None:
        payload = ToolStart(id=call_id, name=name, arguments=arguments)
        asyncio.create_task(self._send(payload))

    def on_tool_result(self, call_id: str, name: str, content: str) -> None:
        payload = ToolResult(id=call_id, name=name, content=content)
        asyncio.create_task(self._send(payload))
