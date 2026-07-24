from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import uuid4

from coding_assistant.core.session_updates import (
    MessageAddedUpdate,
    MessageDeltaUpdate,
    SessionUpdate,
)
from coding_assistant.llm.types import AssistantMessage
from coding_assistant.manager.service import WorkerPrompt, WorkerRunResult


@dataclass
class FakeWorkerRunner:
    response_text: str = "fake response"
    release: asyncio.Event | None = None
    started: asyncio.Event | None = None
    cancelled_session_ids: list[str] | None = None
    prompts: list[WorkerPrompt] | None = None

    async def run_prompt(
        self,
        *,
        prompt: WorkerPrompt,
        on_update: Callable[[SessionUpdate], Awaitable[None]],
    ) -> WorkerRunResult:
        if self.prompts is None:
            self.prompts = []
        self.prompts.append(prompt)
        if self.started is not None:
            self.started.set()
        message_id = f"msg_{uuid4().hex}"
        await on_update(MessageDeltaUpdate(message_id=message_id, append_text=self.response_text))
        assistant_message = AssistantMessage(content=self.response_text)
        await on_update(MessageAddedUpdate(message_id=message_id, message=assistant_message))
        if self.release is not None:
            await self.release.wait()
        return WorkerRunResult(stop_reason="end_turn")

    async def cancel(self, *, session_id: str) -> None:
        if self.cancelled_session_ids is None:
            self.cancelled_session_ids = []
        self.cancelled_session_ids.append(session_id)
