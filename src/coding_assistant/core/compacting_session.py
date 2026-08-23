from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager, suppress

from coding_assistant.core.agent_session import (
    AgentSessionEvent,
    AgentSessionProtocol,
    PromptContent,
    RunFinishedEvent,
    SessionState,
)
from coding_assistant.llm.types import BaseMessage

COMPACTION_PROMPT = "Immediately compact our conversation so far by using the `compact_conversation` tool."


class AutoCompactingSession:
    """Wraps an AgentSession to automatically compact history when token usage crosses a threshold."""

    def __init__(self, inner: AgentSessionProtocol, *, token_budget: int | None) -> None:
        self._inner = inner
        self._token_budget = token_budget
        self._monitor_ready = asyncio.Event()
        self._monitor_task = asyncio.create_task(self._monitor_runs())

    @property
    def token_budget(self) -> int | None:
        """Return the configured token budget threshold, if any."""
        return self._token_budget

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self._inner.model

    @property
    def state(self) -> SessionState:
        """Return the current run state and pending prompt count."""
        return self._inner.state

    @property
    def history(self) -> list[BaseMessage]:
        """Return a copy of the committed transcript history."""
        return self._inner.history

    def subscribe(self) -> AbstractAsyncContextManager[asyncio.Queue[AgentSessionEvent]]:
        """Subscribe to streamed session events."""
        return self._inner.subscribe()

    async def enqueue_prompt(self, content: PromptContent) -> bool:
        """Queue one prompt."""
        await self._monitor_ready.wait()
        return await self._inner.enqueue_prompt(content)

    async def enqueue_prompt_if_idle(self, content: PromptContent) -> bool:
        """Queue one prompt only when idle."""
        await self._monitor_ready.wait()
        return await self._inner.enqueue_prompt_if_idle(content)

    async def enqueue_steering_prompt(self, content: PromptContent) -> bool:
        """Inject one prompt into the active run at the next agent-loop boundary."""
        await self._monitor_ready.wait()
        return await self._inner.enqueue_steering_prompt(content)

    async def pop_last_queued_prompt(self) -> PromptContent | None:
        """Remove and return the last queued prompt, or None if the queue is empty."""
        return await self._inner.pop_last_queued_prompt()

    async def cancel_current_run(self, *, pause_queue: bool = False) -> bool:
        """Cancel the active run, optionally pausing the queue."""
        return await self._inner.cancel_current_run(pause_queue=pause_queue)

    async def resume(self) -> bool:
        """Resume consuming queued prompts after an explicit pause."""
        return await self._inner.resume()

    async def close(self) -> None:
        """Stop the session loop and cancel any active or queued work."""
        self._monitor_task.cancel()
        with suppress(asyncio.CancelledError):
            await self._monitor_task
        await self._inner.close()

    def _should_compact(self) -> bool:
        if self._token_budget is None:
            return False
        state = self._inner.state
        if COMPACTION_PROMPT in state.pending_prompts:
            return False
        usage = state.usage
        return usage is not None and usage.tokens is not None and usage.tokens >= self._token_budget

    async def _monitor_runs(self) -> None:
        """Request compaction after each completed over-budget run."""
        try:
            async with self._inner.subscribe() as events:
                self._monitor_ready.set()
                while True:
                    event = await events.get()
                    if isinstance(event, RunFinishedEvent) and self._should_compact():
                        await self._inner.enqueue_steering_prompt(COMPACTION_PROMPT)
        finally:
            self._monitor_ready.set()
