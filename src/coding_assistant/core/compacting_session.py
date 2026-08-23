from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager

from coding_assistant.core.agent_session import (
    AgentSessionEvent,
    AgentSessionProtocol,
    PromptContent,
    RunOutcome,
    ScheduledRun,
    SessionState,
)
from coding_assistant.llm.types import BaseMessage

COMPACTION_PROMPT = "Immediately compact our conversation so far by using the `compact_conversation` tool."


class AutoCompactingSession:
    """Wraps an AgentSession to automatically compact history when token usage crosses a threshold."""

    def __init__(self, inner: AgentSessionProtocol, *, token_budget: int | None) -> None:
        self._inner = inner
        self._token_budget = token_budget
        self._compaction_scheduled = False

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

    async def enqueue_prompt(self, content: PromptContent) -> ScheduledRun | None:
        """Queue one prompt, triggering compaction first if usage exceeds budget."""
        if self._should_compact() and content != COMPACTION_PROMPT:
            self._compaction_scheduled = True
            compaction_run = await self._inner.enqueue_prompt(COMPACTION_PROMPT)
            if compaction_run is not None:
                compaction_run.completion.add_done_callback(self._on_compaction_finished)
            else:
                self._compaction_scheduled = False
        return await self._inner.enqueue_prompt(content)

    async def enqueue_prompt_if_idle(self, content: PromptContent) -> ScheduledRun | None:
        """Queue one prompt only when idle, triggering compaction first if usage exceeds budget."""
        if self._should_compact() and content != COMPACTION_PROMPT:
            state = self._inner.state
            if state.running or state.paused or state.pending_prompts:
                return None
            self._compaction_scheduled = True
            compaction_run = await self._inner.enqueue_prompt(COMPACTION_PROMPT)
            if compaction_run is not None:
                compaction_run.completion.add_done_callback(self._on_compaction_finished)
            else:
                self._compaction_scheduled = False
            return await self._inner.enqueue_prompt(content)
        return await self._inner.enqueue_prompt_if_idle(content)

    async def enqueue_steering_prompt(self, content: PromptContent) -> bool:
        """Inject one prompt into the active run at the next agent-loop boundary."""
        return await self._inner.enqueue_steering_prompt(content)

    async def pop_last_queued_prompt(self) -> PromptContent | None:
        """Remove and return the last queued prompt, or None if the queue is empty.

        If popping the prompt leaves an auto-scheduled compaction prompt orphaned,
        pops and cancels that compaction prompt as well.
        """
        popped = await self._inner.pop_last_queued_prompt()
        if popped is None:
            return None

        state = self._inner.state
        if self._compaction_scheduled and state.pending_prompts == (COMPACTION_PROMPT,):
            await self._inner.pop_last_queued_prompt()
            self._compaction_scheduled = False

        return popped

    async def cancel_current_run(self, *, pause_queue: bool = False) -> bool:
        """Cancel the active run, optionally pausing the queue."""
        return await self._inner.cancel_current_run(pause_queue=pause_queue)

    async def resume(self) -> bool:
        """Resume consuming queued prompts after an explicit pause."""
        return await self._inner.resume()

    async def close(self) -> None:
        """Stop the session loop and cancel any active or queued work."""
        self._compaction_scheduled = False
        await self._inner.close()

    def _should_compact(self) -> bool:
        if self._token_budget is None or self._compaction_scheduled:
            return False
        state = self._inner.state
        if COMPACTION_PROMPT in state.pending_prompts:
            return False
        usage = state.usage
        return usage is not None and usage.tokens is not None and usage.tokens >= self._token_budget

    def _on_compaction_finished(self, _future: asyncio.Future[RunOutcome]) -> None:
        self._compaction_scheduled = False
