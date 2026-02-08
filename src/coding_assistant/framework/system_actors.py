from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Sequence

from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.execution import AgentActor, ToolCallActor
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks, Tool
from coding_assistant.ui import ActorUI, UI, UserActor


@dataclass(slots=True)
class SystemActors:
    agent_actor: AgentActor
    tool_call_actor: ToolCallActor
    user_actor: UI
    _owns_user_actor: bool = True

    def start(self) -> None:
        if self._owns_user_actor and isinstance(self.user_actor, ActorUI):
            self.user_actor.start()
        self.agent_actor.start()
        self.tool_call_actor.start()

    async def stop(self) -> None:
        await self.tool_call_actor.stop()
        await self.agent_actor.stop()
        if self._owns_user_actor and isinstance(self.user_actor, ActorUI):
            await self.user_actor.stop()


@asynccontextmanager
async def system_actor_scope(
    *,
    tools: Sequence[Tool],
    ui: UI,
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    context_name: str,
) -> AsyncIterator[SystemActors]:
    owns_user_actor = not isinstance(ui, ActorUI)
    user_actor = ui if isinstance(ui, ActorUI) else UserActor(ui, context_name=context_name)
    actors = SystemActors(
        agent_actor=AgentActor(context_name=context_name),
        tool_call_actor=ToolCallActor(
            tools=tools,
            ui=user_actor,
            context_name=context_name,
            progress_callbacks=progress_callbacks,
            tool_callbacks=tool_callbacks,
        ),
        user_actor=user_actor,
        _owns_user_actor=owns_user_actor,
    )
    actors.start()
    try:
        yield actors
    finally:
        await actors.stop()
