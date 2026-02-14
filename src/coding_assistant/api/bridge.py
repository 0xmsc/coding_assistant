import asyncio
import uuid
from typing import Any, Optional

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
from coding_assistant.llm.types import (
    AssistantMessage,
    ProgressCallbacks,
    StatusLevel,
    ToolCall,
    ToolMessage,
    UserMessage,
)


class WebSocketUI(UI):
    def __init__(self, websocket: WebSocket, response_queue: "asyncio.Queue[Any]"):
        self.websocket = websocket
        self.response_queue = response_queue

    async def _send_and_wait(self, request: AskRequest | ConfirmRequest) -> Any:
        envelope = Envelope(payload=request)
        await self.websocket.send_text(envelope.model_dump_json())

        while True:
            response = await self.response_queue.get()
            if response.request_id == request.request_id:
                return response

    async def ask(self, prompt_text: str, default: Optional[str] = None) -> str:
        request_id = str(uuid.uuid4())
        request = AskRequest(request_id=request_id, prompt=prompt_text, default=default)
        response = await self._send_and_wait(request)
        return str(response.text)

    async def confirm(self, prompt_text: str) -> bool:
        request_id = str(uuid.uuid4())
        request = ConfirmRequest(request_id=request_id, prompt=prompt_text)
        response = await self._send_and_wait(request)
        return bool(response.value)

    async def prompt(self, words: list[str] | None = None) -> str:
        return await self.ask("General prompt requested")


class WebSocketProgressCallbacks(ProgressCallbacks):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def _send(self, payload: Any) -> None:
        envelope = Envelope(payload=payload)
        await self.websocket.send_text(envelope.model_dump_json())

    def on_status_message(self, message: str, level: StatusLevel = StatusLevel.INFO) -> None:
        payload = StatusMessage(level=level.value, message=message)
        asyncio.create_task(self._send(payload))

    def on_content_chunk(self, chunk: str) -> None:
        payload = ContentChunk(content=chunk)
        asyncio.create_task(self._send(payload))

    def on_tool_start(self, context_name: str, tool_call: ToolCall, arguments: dict[str, Any]) -> None:
        payload = ToolStart(id=tool_call.id, name=tool_call.function.name, arguments=arguments)
        asyncio.create_task(self._send(payload))

    def on_tool_message(
        self, context_name: str, message: ToolMessage, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        payload = ToolResult(id=message.tool_call_id, name=tool_name, content=message.content)
        asyncio.create_task(self._send(payload))

    def on_user_message(self, context_name: str, message: UserMessage, *, force: bool = False) -> None:
        pass

    def on_assistant_message(self, context_name: str, message: AssistantMessage, *, force: bool = False) -> None:
        pass

    def on_reasoning_chunk(self, chunk: str) -> None:
        pass

    def on_chunks_end(self) -> None:
        pass
