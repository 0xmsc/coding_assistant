from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from coding_assistant.core.session_updates import AgentMessageChunkUpdate, SessionUpdate
from coding_assistant.llm.types import AssistantMessage, UserMessage
from coding_assistant.manager.service import WorkerCommit, WorkerPrompt
from coding_assistant.remote.acp import prompt_content_from_acp


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
    ) -> WorkerCommit:
        if self.prompts is None:
            self.prompts = []
        self.prompts.append(prompt)
        if self.started is not None:
            self.started.set()
        await on_update(AgentMessageChunkUpdate(content=self.response_text))
        if self.release is not None:
            await self.release.wait()
        return WorkerCommit(
            messages=[
                UserMessage(content=prompt_content_from_acp(prompt.prompt)),
                AssistantMessage(content=self.response_text),
            ],
            stop_reason="end_turn",
        )

    async def cancel(self, *, session_id: str) -> None:
        if self.cancelled_session_ids is None:
            self.cancelled_session_ids = []
        self.cancelled_session_ids.append(session_id)
