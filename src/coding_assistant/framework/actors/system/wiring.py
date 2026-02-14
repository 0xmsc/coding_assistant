from __future__ import annotations

from dataclasses import dataclass

from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actors.agent.actor import AgentActor
from coding_assistant.framework.actors.chat.actor import ChatActor
from coding_assistant.framework.actors.common.messages import ToolCapability
from coding_assistant.framework.actors.llm.actor import LLMActor
from coding_assistant.framework.actors.tool_call.actor import ToolCallActor
from coding_assistant.framework.actors.tool_call.capabilities import ToolCapabilityActor, register_tool_capabilities
from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.callbacks import ToolCallbacks
from coding_assistant.framework.types import Completer
from coding_assistant.llm.types import ProgressCallbacks, Tool
from coding_assistant.ui import UI, UserActor


@dataclass(slots=True)
class OrchestratorActors:
    actor_directory: ActorDirectory
    agent_actor: AgentActor
    chat_actor: ChatActor
    llm_actor: LLMActor
    tool_call_actor: ToolCallActor
    user_actor: UI
    tool_capabilities: tuple[ToolCapability, ...]
    tool_capability_actors: tuple[ToolCapabilityActor, ...]
    agent_actor_uri: str
    chat_actor_uri: str
    llm_actor_uri: str
    tool_call_actor_uri: str
    user_actor_uri: str


def build_orchestrator_actors(
    *,
    ui: UI,
    tools: list[Tool],
    callbacks: ProgressCallbacks,
    tool_callbacks: ToolCallbacks,
    completer: Completer,
    context_name: str = "Orchestrator",
) -> OrchestratorActors:
    actor_directory = ActorDirectory()
    agent_actor_uri = "actor://orchestrator/agent"
    chat_actor_uri = "actor://orchestrator/chat"
    llm_actor_uri = "actor://orchestrator/llm"
    tool_call_actor_uri = "actor://orchestrator/tool-call"
    user_actor_uri = "actor://orchestrator/user"

    user_actor = UserActor(ui, context_name=context_name, actor_directory=actor_directory)

    tool_call_tools = list(tools)
    if not any(tool.name() == "compact_conversation" for tool in tool_call_tools):
        tool_call_tools.append(CompactConversationTool())
    tool_capabilities, tool_capability_actors = register_tool_capabilities(
        actor_directory=actor_directory,
        tools=tuple(tool_call_tools),
        context_name=context_name,
        uri_prefix="actor://orchestrator/tool-capability",
    )
    tool_call_actor = ToolCallActor(
        default_tool_capabilities=tool_capabilities,
        ui=ui,
        context_name=context_name,
        actor_directory=actor_directory,
        progress_callbacks=callbacks,
        tool_callbacks=tool_callbacks,
    )
    llm_actor = LLMActor(
        completer=completer,
        progress_callbacks=callbacks,
        context_name=context_name,
        actor_directory=actor_directory,
    )
    chat_actor = ChatActor(
        actor_directory=actor_directory,
        self_uri=chat_actor_uri,
        llm_actor_uri=llm_actor_uri,
        progress_callbacks=callbacks,
        context_name=context_name,
    )
    agent_actor = AgentActor(
        actor_directory=actor_directory,
        self_uri=agent_actor_uri,
        llm_actor_uri=llm_actor_uri,
        progress_callbacks=callbacks,
        context_name=context_name,
    )
    actor_directory.register(uri=agent_actor_uri, actor=agent_actor)
    actor_directory.register(uri=chat_actor_uri, actor=chat_actor)
    actor_directory.register(uri=llm_actor_uri, actor=llm_actor)
    actor_directory.register(uri=tool_call_actor_uri, actor=tool_call_actor)
    actor_directory.register(uri=user_actor_uri, actor=user_actor)
    return OrchestratorActors(
        actor_directory=actor_directory,
        agent_actor=agent_actor,
        chat_actor=chat_actor,
        llm_actor=llm_actor,
        tool_call_actor=tool_call_actor,
        user_actor=user_actor,
        tool_capabilities=tool_capabilities,
        tool_capability_actors=tool_capability_actors,
        agent_actor_uri=agent_actor_uri,
        chat_actor_uri=chat_actor_uri,
        llm_actor_uri=llm_actor_uri,
        tool_call_actor_uri=tool_call_actor_uri,
        user_actor_uri=user_actor_uri,
    )


def start_orchestrator_actors(actors: OrchestratorActors) -> None:
    if isinstance(actors.user_actor, UserActor):
        actors.user_actor.start()
    actors.llm_actor.start()
    actors.tool_call_actor.start()
    actors.agent_actor.start()
    actors.chat_actor.start()


async def shutdown_orchestrator_actors(actors: OrchestratorActors) -> None:
    await actors.tool_call_actor.stop()
    await actors.chat_actor.stop()
    await actors.agent_actor.stop()
    await actors.llm_actor.stop()
    for capability_actor in actors.tool_capability_actors:
        await capability_actor.stop()
    for capability in actors.tool_capabilities:
        actors.actor_directory.unregister(uri=capability.uri)
    if isinstance(actors.user_actor, UserActor):
        await actors.user_actor.stop()
    actors.actor_directory.unregister(uri=actors.agent_actor_uri)
    actors.actor_directory.unregister(uri=actors.chat_actor_uri)
    actors.actor_directory.unregister(uri=actors.llm_actor_uri)
    actors.actor_directory.unregister(uri=actors.tool_call_actor_uri)
    actors.actor_directory.unregister(uri=actors.user_actor_uri)
