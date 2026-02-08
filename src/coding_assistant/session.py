import logging
from pathlib import Path
from typing import Any, Optional

from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.framework.builtin_tools import CompactConversationTool
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks, StatusLevel
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.execution import AgentActor, ToolCallActor
from coding_assistant.framework.system_actors import SystemActors
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
        self._system_actors: SystemActors | None = None

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

        user_actor = UserActor(self.ui, context_name="Orchestrator")
        tool_call_tools = list(self.tools)
        if not any(tool.name() == "compact_conversation" for tool in tool_call_tools):
            tool_call_tools.append(CompactConversationTool())
        self._system_actors = SystemActors(
            agent_actor=AgentActor(context_name="Orchestrator"),
            tool_call_actor=ToolCallActor(
                tools=tool_call_tools,
                ui=user_actor,
                context_name="Orchestrator",
                progress_callbacks=self.callbacks,
                tool_callbacks=self.tool_callbacks,
            ),
            user_actor=user_actor,
        )
        self._system_actors.start()

        self.callbacks.on_status_message("Session initialized.", level=StatusLevel.SUCCESS)
        self.callbacks.on_status_message(f"Using model {self.config.model}.", level=StatusLevel.SUCCESS)

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._system_actors:
            await self._system_actors.stop()
            self._system_actors = None

        if self._mcp_manager:
            await self._mcp_manager.shutdown(exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb)

        self.callbacks.on_status_message("Session closed.", level=StatusLevel.INFO)

    async def run_chat(self, history: Optional[list[BaseMessage]] = None) -> None:
        chat_history = history or []
        async with history_manager_scope(context_name="session") as history_manager:
            try:
                ui = self._system_actors.user_actor if self._system_actors else self.ui
                await run_chat_loop(
                    history=chat_history,
                    model=self.config.model,
                    tools=self.tools,
                    instructions=self.instructions,
                    callbacks=self.callbacks,
                    tool_callbacks=self.tool_callbacks,
                    completer=openai_complete,
                    ui=ui,
                    context_name="Orchestrator",
                    system_actors=self._system_actors,
                )
            finally:
                await history_manager.save_orchestrator_history(
                    working_directory=self.working_directory, history=chat_history
                )

    async def run_agent(self, task: str, history: Optional[list[BaseMessage]] = None) -> Any:
        ui = self._system_actors.user_actor if self._system_actors else self.ui
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
