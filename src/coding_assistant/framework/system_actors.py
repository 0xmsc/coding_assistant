from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Sequence, TYPE_CHECKING

from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.execution import AgentActor, ToolCallActor
from coding_assistant.framework.types import AgentContext, Completer
from coding_assistant.llm.types import BaseMessage, NullProgressCallbacks, ProgressCallbacks, Tool
from coding_assistant.ui import ActorUI, UI, UserActor

if TYPE_CHECKING:
    from coding_assistant.framework.chat import ChatLoopActor


@dataclass(slots=True)
class SystemActors:
    agent_actor: AgentActor
    tool_call_actor: ToolCallActor
    chat_actor: ChatLoopActor
    user_actor: UI
    _owns_user_actor: bool = True

    def start(self) -> None:
        if self._owns_user_actor and isinstance(self.user_actor, ActorUI):
            self.user_actor.start()
        self.agent_actor.start()
        self.tool_call_actor.start()
        self.chat_actor.start()

    async def stop(self) -> None:
        await self.chat_actor.stop()
        await self.tool_call_actor.stop()
        await self.agent_actor.stop()
        if self._owns_user_actor and isinstance(self.user_actor, ActorUI):
            await self.user_actor.stop()

    async def run_chat_loop(
        self,
        *,
        history: list[BaseMessage],
        model: str,
        tools: list[Tool],
        instructions: str | None,
        callbacks: ProgressCallbacks,
        completer: Completer,
        context_name: str,
    ) -> None:
        await self.chat_actor.run_chat_loop(
            history=history,
            model=model,
            tools=tools,
            instructions=instructions,
            callbacks=callbacks,
            completer=completer,
            context_name=context_name,
        )

    async def run_agent_loop(
        self,
        ctx: AgentContext,
        *,
        tools: Sequence[Tool],
        progress_callbacks: ProgressCallbacks,
        completer: Completer,
        compact_conversation_at_tokens: int,
    ) -> None:
        await self.agent_actor.run_agent_loop(
            ctx,
            tools=tools,
            progress_callbacks=progress_callbacks,
            completer=completer,
            compact_conversation_at_tokens=compact_conversation_at_tokens,
            tool_call_actor=self.tool_call_actor,
        )


@asynccontextmanager
async def system_actor_scope(
    *,
    tools: Sequence[Tool],
    ui: UI,
    progress_callbacks: ProgressCallbacks = NullProgressCallbacks(),
    tool_callbacks: ToolCallbacks = NullToolCallbacks(),
    context_name: str,
) -> AsyncIterator[SystemActors]:
    from coding_assistant.framework.chat import ChatLoopActor

    owns_user_actor = not isinstance(ui, ActorUI)
    user_actor = ui if isinstance(ui, ActorUI) else UserActor(ui, context_name=context_name)
    tool_call_actor = ToolCallActor(
        tools=tools,
        ui=user_actor,
        context_name=context_name,
        progress_callbacks=progress_callbacks,
        tool_callbacks=tool_callbacks,
    )
    agent_actor = AgentActor(context_name=context_name)
    actors = SystemActors(
        agent_actor=agent_actor,
        tool_call_actor=tool_call_actor,
        chat_actor=ChatLoopActor(
            user_actor=user_actor,
            tool_call_actor=tool_call_actor,
            agent_actor=agent_actor,
            context_name=context_name,
        ),
        user_actor=user_actor,
        _owns_user_actor=owns_user_actor,
    )
    actors.start()
    try:
        yield actors
    finally:
        await actors.stop()
