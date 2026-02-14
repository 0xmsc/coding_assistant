import logging
import asyncio
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.framework.actor_directory import ActorDirectory
from coding_assistant.framework.actors.agent.actor import AgentActor
from coding_assistant.framework.actors.chat.actor import ChatActor
from coding_assistant.framework.actors.common.messages import RunChatRequest, RunCompleted, RunFailed
from coding_assistant.framework.actors.llm.actor import LLMActor
from coding_assistant.framework.actors.tool_call.actor import ToolCallActor
from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks, StatusLevel
from coding_assistant.llm.types import BaseMessage, Tool
from coding_assistant.history_manager import history_manager_scope
from coding_assistant.instructions import get_instructions
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.sandbox import sandbox
from coding_assistant.tools.mcp_manager import MCPServerManager
from coding_assistant.tools.tools import AgentTool, AskClientTool, RedirectToolCallTool
from coding_assistant.ui import UI, UserActor

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
        self._agent_actor_uri: str | None = None
        self._chat_actor_uri: str | None = None
        self._llm_actor_uri: str | None = None
        self._tool_call_actor_uri: str | None = None
        self._user_actor_uri: str | None = None

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

        actor_directory = ActorDirectory()
        agent_actor_uri = "actor://orchestrator/agent"
        chat_actor_uri = "actor://orchestrator/chat"
        llm_actor_uri = "actor://orchestrator/llm"
        tool_call_actor_uri = "actor://orchestrator/tool-call"
        user_actor_uri = "actor://orchestrator/user"

        user_actor = UserActor(self.ui, context_name="Orchestrator", actor_directory=actor_directory)

        tool_call_tools = list(self.tools)
        if not any(tool.name() == "compact_conversation" for tool in tool_call_tools):
            tool_call_tools.append(CompactConversationTool())
        tool_call_actor = ToolCallActor(
            tools=tool_call_tools,
            ui=self.ui,
            context_name="Orchestrator",
            actor_directory=actor_directory,
            progress_callbacks=self.callbacks,
            tool_callbacks=self.tool_callbacks,
        )
        llm_actor = LLMActor(context_name="Orchestrator", actor_directory=actor_directory)
        chat_actor = ChatActor(
            actor_directory=actor_directory,
            self_uri=chat_actor_uri,
            llm_actor_uri=llm_actor_uri,
            context_name="Orchestrator",
        )
        agent_actor = AgentActor(
            actor_directory=actor_directory,
            self_uri=agent_actor_uri,
            llm_actor_uri=llm_actor_uri,
            context_name="Orchestrator",
        )
        actor_directory.register(uri=agent_actor_uri, actor=agent_actor)
        actor_directory.register(uri=chat_actor_uri, actor=chat_actor)
        actor_directory.register(uri=llm_actor_uri, actor=llm_actor)
        actor_directory.register(uri=tool_call_actor_uri, actor=tool_call_actor)
        actor_directory.register(uri=user_actor_uri, actor=user_actor)
        self._actor_directory = actor_directory
        self._agent_actor = agent_actor
        self._chat_actor = chat_actor
        self._llm_actor = llm_actor
        self._tool_call_actor = tool_call_actor
        self._user_actor = user_actor
        self._agent_actor_uri = agent_actor_uri
        self._chat_actor_uri = chat_actor_uri
        self._llm_actor_uri = llm_actor_uri
        self._tool_call_actor_uri = tool_call_actor_uri
        self._user_actor_uri = user_actor_uri
        if isinstance(user_actor, UserActor):
            user_actor.start()
        self._llm_actor.start()
        self._tool_call_actor.start()
        self._agent_actor.start()
        self._chat_actor.start()

        self.callbacks.on_status_message("Session initialized.", level=StatusLevel.SUCCESS)
        self.callbacks.on_status_message(f"Using model {self.config.model}.", level=StatusLevel.SUCCESS)

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._tool_call_actor:
            await self._tool_call_actor.stop()
            self._tool_call_actor = None
        if self._chat_actor:
            await self._chat_actor.stop()
            self._chat_actor = None
        if self._agent_actor:
            await self._agent_actor.stop()
            self._agent_actor = None
        if self._llm_actor:
            await self._llm_actor.stop()
            self._llm_actor = None
        if self._user_actor and isinstance(self._user_actor, UserActor):
            await self._user_actor.stop()
        self._user_actor = None
        if self._actor_directory is not None:
            if self._agent_actor_uri is not None:
                self._actor_directory.unregister(uri=self._agent_actor_uri)
            if self._chat_actor_uri is not None:
                self._actor_directory.unregister(uri=self._chat_actor_uri)
            if self._llm_actor_uri is not None:
                self._actor_directory.unregister(uri=self._llm_actor_uri)
            if self._tool_call_actor_uri is not None:
                self._actor_directory.unregister(uri=self._tool_call_actor_uri)
            if self._user_actor_uri is not None:
                self._actor_directory.unregister(uri=self._user_actor_uri)
        self._actor_directory = None
        self._agent_actor_uri = None
        self._chat_actor_uri = None
        self._llm_actor_uri = None
        self._tool_call_actor_uri = None
        self._user_actor_uri = None

        if self._mcp_manager:
            await self._mcp_manager.shutdown(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

        self.callbacks.on_status_message("Session closed.", level=StatusLevel.INFO)

    async def run_chat(self, history: Optional[list[BaseMessage]] = None) -> None:
        chat_history = history or []
        if (
            self._actor_directory is None
            or self._chat_actor_uri is None
            or self._tool_call_actor_uri is None
            or self._user_actor_uri is None
        ):
            raise RuntimeError("Session actors are not initialized. Use `async with Session(...)` before running chat.")
        tools_with_meta = list(self.tools)
        if not any(tool.name() == "compact_conversation" for tool in tools_with_meta):
            tools_with_meta.append(CompactConversationTool())
        async with history_manager_scope(context_name="session") as history_manager:
            request_id = uuid4().hex
            reply_uri = f"actor://session/chat-reply/{request_id}"
            completion_future = self._register_run_reply(request_id=request_id, reply_uri=reply_uri)
            try:
                await self._actor_directory.send_message(
                    uri=self._chat_actor_uri,
                    message=RunChatRequest(
                        request_id=request_id,
                        history=chat_history,
                        model=self.config.model,
                        tools=tuple(tools_with_meta),
                        instructions=self.instructions,
                        context_name="Orchestrator",
                        callbacks=self.callbacks,
                        completer=openai_complete,
                        user_actor_uri=self._user_actor_uri,
                        tool_call_actor_uri=self._tool_call_actor_uri,
                        reply_to_uri=reply_uri,
                    ),
                )
                await completion_future
            finally:
                self._actor_directory.unregister(uri=reply_uri)
                await history_manager.save_orchestrator_history(
                    working_directory=self.working_directory, history=chat_history
                )

    async def run_agent(self, task: str, history: Optional[list[BaseMessage]] = None) -> Any:
        if self._actor_directory is None or self._agent_actor_uri is None or self._tool_call_actor_uri is None:
            raise RuntimeError(
                "Session actors are not initialized. Use `async with Session(...)` before running agent."
            )
        ui = self._user_actor if self._user_actor else self.ui
        agent_mode_tools = [
            AskClientTool(ui=ui),
            *self.tools,
        ]

        tool = AgentTool(
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

        params = {
            "task": task,
            "instructions": self.instructions,
            "expert_knowledge": True,
        }

        async with history_manager_scope(context_name="session") as history_manager:
            try:
                result = await tool.execute(params)
                self.callbacks.on_status_message(f"Task completed: {result.content}", level=StatusLevel.SUCCESS)
                return result
            finally:
                await history_manager.save_orchestrator_history(
                    working_directory=self.working_directory, history=tool.history
                )

    def _register_run_reply(self, *, request_id: str, reply_uri: str) -> "asyncio.Future[None]":
        if self._actor_directory is None:
            raise RuntimeError("Session actors are not initialized. Use `async with Session(...)` before running.")

        completion_future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        reply_actor = _RunReplyActor(request_id=request_id, future=completion_future)
        self._actor_directory.register(uri=reply_uri, actor=reply_actor)
        return completion_future


class _RunReplyActor:
    def __init__(self, *, request_id: str, future: "asyncio.Future[None]") -> None:
        self._request_id = request_id
        self._future = future

    async def send_message(self, message: object) -> None:
        if isinstance(message, RunCompleted):
            if message.request_id != self._request_id:
                self._future.set_exception(RuntimeError(f"Mismatched run response id: {message.request_id}"))
                return
            self._future.set_result(None)
            return
        if isinstance(message, RunFailed):
            if message.request_id != self._request_id:
                self._future.set_exception(RuntimeError(f"Mismatched run response id: {message.request_id}"))
                return
            self._future.set_exception(message.error)
            return
        self._future.set_exception(RuntimeError(f"Unexpected run response type: {type(message).__name__}"))
