import asyncio
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actors.agent.actor import AgentActor
from coding_assistant.framework.actors.chat.actor import ChatActor
from coding_assistant.framework.actors.common.messages import RunAgentRequest, RunChatRequest, ToolCapability
from coding_assistant.framework.actors.common.messages import RunCompleted
from coding_assistant.framework.actors.common.reply_waiters import (
    register_run_payload_reply_waiter,
    register_run_reply_waiter,
)
from coding_assistant.framework.actors.llm.actor import LLMActor
from coding_assistant.framework.actors.system.wiring import (
    OrchestratorActors,
    build_orchestrator_actors,
    shutdown_orchestrator_actors,
    start_orchestrator_actors,
)
from coding_assistant.framework.actors.tool_call.capabilities import ToolCapabilityActor, register_tool_capabilities
from coding_assistant.framework.actors.tool_call.actor import ToolCallActor
from coding_assistant.framework.builtin_tools import FinishTaskTool
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.framework.parameters import Parameter, parameters_from_model
from coding_assistant.framework.results import TextResult
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks, StatusLevel
from coding_assistant.llm.types import BaseMessage, Tool
from coding_assistant.history_manager import history_manager_scope
from coding_assistant.instructions import get_instructions
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.sandbox import sandbox
from coding_assistant.tools.mcp_manager import MCPServerManager
from coding_assistant.tools.tools import AgentTool, AskClientTool, LaunchAgentSchema, RedirectToolCallTool
from coding_assistant.ui import UI

logger = logging.getLogger(__name__)


class Session:
    def __init__(
        self,
        *,
        config: Config,
        ui: UI,
        callbacks: ProgressCallbacks = NullProgressCallbacks(),
        tool_callbacks: ToolCallbacks = NullToolCallbacks(),
        working_directory: Path,
        coding_assistant_root: Path,
        mcp_server_configs: list[MCPServerConfig],
        skills_directories: Optional[list[str]] = None,
        mcp_env: Optional[list[str]] = None,
        sandbox_enabled: bool = True,
        readable_sandbox_directories: Optional[list[Path]] = None,
        writable_sandbox_directories: Optional[list[Path]] = None,
        user_instructions: Optional[list[str]] = None,
    ):
        self.config = config
        self.ui = ui
        self.callbacks = callbacks
        self.tool_callbacks = tool_callbacks
        self.working_directory = working_directory
        self.coding_assistant_root = coding_assistant_root
        self.skills_directories = skills_directories or []
        self.mcp_env_list = mcp_env or []
        self.sandbox_enabled = sandbox_enabled
        self.readable_sandbox_directories = readable_sandbox_directories or []
        self.writable_sandbox_directories = writable_sandbox_directories or []
        self.user_instructions = user_instructions or []

        # Build initial list of server configs
        self.mcp_server_configs = list(mcp_server_configs)

        self.tools: list[Tool] = []
        self.instructions: str = ""
        self._mcp_manager: Optional[MCPServerManager] = None
        self._mcp_servers: Optional[list[Any]] = None
        self._agent_actor: AgentActor | None = None
        self._llm_actor: LLMActor | None = None
        self._tool_call_actor: ToolCallActor | None = None
        self._chat_actor: ChatActor | None = None
        self._user_actor: UI | None = None
        self._actor_directory: ActorDirectory | None = None
        self._orchestrator_actors: OrchestratorActors | None = None
        self._agent_actor_uri: str | None = None
        self._chat_actor_uri: str | None = None
        self._llm_actor_uri: str | None = None
        self._tool_call_actor_uri: str | None = None
        self._user_actor_uri: str | None = None
        self._tool_capabilities: tuple[ToolCapability, ...] = tuple()

    @property
    def mcp_servers(self) -> Optional[list[Any]]:
        return self._mcp_servers

    @staticmethod
    def get_default_mcp_server_config(
        root_directory: Path, skills_directories: list[str], env: list[str] | None = None
    ) -> MCPServerConfig:
        import sys

        args = [
            "-m",
            "coding_assistant.mcp",
        ]

        if skills_directories:
            args.append("--skills-directories")
            args.extend(skills_directories)

        return MCPServerConfig(
            name="coding_assistant.mcp",
            command=sys.executable,
            args=args,
            env=env or [],
        )

    async def __aenter__(self) -> "Session":
        self.callbacks.on_status_message("Initializing session...", level=StatusLevel.INFO)

        # Sandbox setup
        if self.sandbox_enabled:
            readable = [
                *self.readable_sandbox_directories,
                *[Path(d).resolve() for d in self.skills_directories],
                self.coding_assistant_root,
            ]
            writable = [*self.writable_sandbox_directories, self.working_directory]
            sandbox(readable_paths=readable, writable_paths=writable, include_defaults=True)
            self.callbacks.on_status_message("Sandboxing enabled.", level=StatusLevel.INFO)

        # Build default server config
        default_config = self.get_default_mcp_server_config(
            self.coding_assistant_root,
            self.skills_directories,
            env=self.mcp_env_list,
        )
        all_configs = [*self.mcp_server_configs, default_config]

        # MCP Servers setup
        self._mcp_manager = MCPServerManager(context_name="session")
        self._mcp_manager.start()
        bundle = await self._mcp_manager.initialize(
            config_servers=all_configs, working_directory=self.working_directory
        )
        self._mcp_servers = bundle.servers
        self.tools = bundle.tools

        # Meta tools
        self.tools.append(RedirectToolCallTool(tools=self.tools))

        # Instructions setup
        self.instructions = get_instructions(
            working_directory=self.working_directory,
            user_instructions=self.user_instructions,
            mcp_servers=self._mcp_servers,
        )

        orchestrator_actors = build_orchestrator_actors(
            ui=self.ui,
            tools=self.tools,
            callbacks=self.callbacks,
            tool_callbacks=self.tool_callbacks,
            completer=openai_complete,
            context_name="Orchestrator",
        )
        self._orchestrator_actors = orchestrator_actors
        self._actor_directory = orchestrator_actors.actor_directory
        self._agent_actor = orchestrator_actors.agent_actor
        self._chat_actor = orchestrator_actors.chat_actor
        self._llm_actor = orchestrator_actors.llm_actor
        self._tool_call_actor = orchestrator_actors.tool_call_actor
        self._user_actor = orchestrator_actors.user_actor
        self._agent_actor_uri = orchestrator_actors.agent_actor_uri
        self._chat_actor_uri = orchestrator_actors.chat_actor_uri
        self._llm_actor_uri = orchestrator_actors.llm_actor_uri
        self._tool_call_actor_uri = orchestrator_actors.tool_call_actor_uri
        self._user_actor_uri = orchestrator_actors.user_actor_uri
        self._tool_capabilities = orchestrator_actors.tool_capabilities
        start_orchestrator_actors(orchestrator_actors)

        self.callbacks.on_status_message("Session initialized.", level=StatusLevel.SUCCESS)
        self.callbacks.on_status_message(f"Using model {self.config.model}.", level=StatusLevel.SUCCESS)

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._orchestrator_actors is not None:
            await shutdown_orchestrator_actors(self._orchestrator_actors)
            self._orchestrator_actors = None
        self._agent_actor = None
        self._chat_actor = None
        self._llm_actor = None
        self._tool_call_actor = None
        self._user_actor = None
        self._actor_directory = None
        self._agent_actor_uri = None
        self._chat_actor_uri = None
        self._llm_actor_uri = None
        self._tool_call_actor_uri = None
        self._user_actor_uri = None
        self._tool_capabilities = tuple()

        if self._mcp_manager:
            await self._mcp_manager.shutdown(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

        self.callbacks.on_status_message("Session closed.", level=StatusLevel.INFO)

    async def run_chat(self, history: Optional[list[BaseMessage]] = None) -> None:
        chat_history = list(history) if history is not None else []
        if (
            self._actor_directory is None
            or self._chat_actor_uri is None
            or self._tool_call_actor_uri is None
            or self._user_actor_uri is None
        ):
            raise RuntimeError("Session actors are not initialized. Use `async with Session(...)` before running chat.")
        async with history_manager_scope(context_name="session") as history_manager:
            request_id = uuid4().hex
            reply_uri = f"actor://session/chat-reply/{request_id}"
            completion_future = self._register_run_payload_reply(request_id=request_id, reply_uri=reply_uri)
            try:
                await self._actor_directory.send_message(
                    uri=self._chat_actor_uri,
                    message=RunChatRequest(
                        request_id=request_id,
                        history=chat_history,
                        model=self.config.model,
                        tool_capabilities=self._tool_capabilities,
                        instructions=self.instructions,
                        context_name="Orchestrator",
                        user_actor_uri=self._user_actor_uri,
                        tool_call_actor_uri=self._tool_call_actor_uri,
                        reply_to_uri=reply_uri,
                    ),
                )
                completion = await completion_future
            finally:
                self._actor_directory.unregister(uri=reply_uri)
                final_history = (
                    list(completion.history)
                    if "completion" in locals() and completion.history is not None
                    else chat_history
                )
                await history_manager.save_orchestrator_history(
                    working_directory=self.working_directory, history=final_history
                )

    async def run_agent(self, task: str, history: Optional[list[BaseMessage]] = None) -> Any:
        if (
            self._actor_directory is None
            or self._agent_actor_uri is None
            or self._tool_call_actor_uri is None
            or self._user_actor_uri is None
        ):
            raise RuntimeError(
                "Session actors are not initialized. Use `async with Session(...)` before running agent."
            )
        ui = self._user_actor if self._user_actor else self.ui
        agent_mode_tools = [
            AskClientTool(ui=ui),
            *self.tools,
        ]

        launch_agent_tool = AgentTool(
            model=self.config.model,
            expert_model=self.config.expert_model,
            compact_conversation_at_tokens=self.config.compact_conversation_at_tokens,
            enable_ask_user=self.config.enable_ask_user,
            tools=agent_mode_tools,
            history=history,
            progress_callbacks=self.callbacks,
            ui=ui,
            tool_callbacks=self.tool_callbacks,
            name="launch_orchestrator_agent",
            completer=openai_complete,
            actor_directory=self._actor_directory,
            agent_actor_uri=self._agent_actor_uri,
            tool_call_actor_uri=self._tool_call_actor_uri,
            user_actor_uri=self._user_actor_uri,
        )
        validated = LaunchAgentSchema.model_validate(
            {
                "task": task,
                "instructions": self.instructions,
                "expert_knowledge": True,
            }
        )
        desc = AgentDescription(
            name="Agent",
            model=self.config.expert_model,
            parameters=[
                Parameter(
                    name="description",
                    description="The description of the agent's work and capabilities.",
                    value=launch_agent_tool.description(),
                ),
                *parameters_from_model(validated),
            ],
            tools=[launch_agent_tool, *agent_mode_tools],
        )
        state = AgentState(history=list(history) if history is not None else [])
        runtime_tools = (
            launch_agent_tool,
            AskClientTool(ui=ui),
            FinishTaskTool(),
        )
        runtime_tool_capabilities: tuple[ToolCapability, ...] = tuple()
        runtime_tool_actors: tuple[ToolCapabilityActor, ...] = tuple()
        if self._actor_directory is not None:
            runtime_tool_capabilities, runtime_tool_actors = register_tool_capabilities(
                actor_directory=self._actor_directory,
                tools=runtime_tools,
                context_name="session",
                uri_prefix="actor://session/runtime-tool-capability",
            )
        effective_capabilities = (*runtime_tool_capabilities, *self._tool_capabilities)

        try:
            async with history_manager_scope(context_name="session") as history_manager:
                try:
                    request_id = uuid4().hex
                    reply_uri = f"actor://session/agent-reply/{request_id}"
                    completion_future = self._register_run_payload_reply(request_id=request_id, reply_uri=reply_uri)
                    try:
                        await self._actor_directory.send_message(
                            uri=self._agent_actor_uri,
                            message=RunAgentRequest(
                                request_id=request_id,
                                ctx=AgentContext(desc=desc, state=state),
                                tool_capabilities=tuple(effective_capabilities),
                                compact_conversation_at_tokens=self.config.compact_conversation_at_tokens,
                                tool_call_actor_uri=self._tool_call_actor_uri,
                                reply_to_uri=reply_uri,
                            ),
                        )
                        completion = await completion_future
                    finally:
                        self._actor_directory.unregister(uri=reply_uri)

                    final_output = completion.agent_output or state.output
                    if final_output is None:
                        raise RuntimeError("Agent did not produce output.")
                    result = TextResult(content=final_output.result)
                    self.callbacks.on_status_message(f"Task completed: {result.content}", level=StatusLevel.SUCCESS)
                    return result
                finally:
                    final_history = (
                        list(completion.history)
                        if "completion" in locals() and completion.history is not None
                        else state.history
                    )
                    await history_manager.save_orchestrator_history(
                        working_directory=self.working_directory, history=final_history
                    )
        finally:
            for capability_actor in runtime_tool_actors:
                await capability_actor.stop()
            if self._actor_directory is not None:
                for capability in runtime_tool_capabilities:
                    self._actor_directory.unregister(uri=capability.uri)

    def _register_run_reply(self, *, request_id: str, reply_uri: str) -> "asyncio.Future[None]":
        if self._actor_directory is None:
            raise RuntimeError("Session actors are not initialized. Use `async with Session(...)` before running.")
        return register_run_reply_waiter(self._actor_directory, request_id=request_id, reply_uri=reply_uri)

    def _register_run_payload_reply(self, *, request_id: str, reply_uri: str) -> "asyncio.Future[RunCompleted]":
        if self._actor_directory is None:
            raise RuntimeError("Session actors are not initialized. Use `async with Session(...)` before running.")
        return register_run_payload_reply_waiter(self._actor_directory, request_id=request_id, reply_uri=reply_uri)
