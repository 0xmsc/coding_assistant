import logging
import sys
from pathlib import Path
from typing import Any, Optional

from coding_assistant.config import Config, MCPServerConfig
from coding_assistant.framework.callbacks import NullToolCallbacks, ToolCallbacks
from coding_assistant.llm.types import NullProgressCallbacks, ProgressCallbacks, StatusLevel
from coding_assistant.framework.chat import run_chat_loop
from coding_assistant.framework.agent import run_agent_loop
from coding_assistant.framework.types import AgentContext, AgentDescription, AgentState
from coding_assistant.llm.types import BaseMessage, Tool
from coding_assistant.history import save_orchestrator_history
from coding_assistant.instructions import get_instructions
from coding_assistant.llm.openai import complete as openai_complete
from coding_assistant.sandbox import sandbox
from coding_assistant.tools.mcp import get_mcp_servers_from_config, get_mcp_wrapped_tools
from coding_assistant.tools.tools import AskClientTool, RedirectToolCallTool
from coding_assistant.ui import UI
from coding_assistant.actors.system import ActorSystem
from coding_assistant.actors.tool_worker import ToolWorkerActor
from coding_assistant.actors.ui_gateway import UIGatewayActor

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
        actor_system: Optional[ActorSystem] = None,
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

        self.actor_system = actor_system or ActorSystem()

        # Build initial list of server configs
        self.mcp_server_configs = list(mcp_server_configs)

        self.tools: list[Tool] = []
        self.instructions: str = ""
        self._mcp_servers_cm: Optional[Any] = None
        self._mcp_servers: Optional[list[Any]] = None

    @property
    def mcp_servers(self) -> Optional[list[Any]]:
        return self._mcp_servers

    @staticmethod
    def get_default_mcp_server_config(
        root_directory: Path, skills_directories: list[str], env: list[str] | None = None
    ) -> MCPServerConfig:
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
        self._mcp_servers_cm = get_mcp_servers_from_config(all_configs, working_directory=self.working_directory)
        assert self._mcp_servers_cm is not None
        self._mcp_servers = await self._mcp_servers_cm.__aenter__()

        assert self._mcp_servers is not None
        self.tools = await get_mcp_wrapped_tools(self._mcp_servers)

        # Meta tools
        self.tools.append(RedirectToolCallTool(tools=self.tools))

        # Instructions setup
        self.instructions = get_instructions(
            working_directory=self.working_directory,
            user_instructions=self.user_instructions,
            mcp_servers=self._mcp_servers,
        )

        # Actor infrastructure startup
        # Register a shared ToolWorker for global tools
        # (Individual agents can spawn their own later)
        global_tool_worker = ToolWorkerActor("global_tool_worker", self.actor_system, self.tools)
        self.actor_system.register(global_tool_worker)
        await global_tool_worker.start()

        # Register UI Gateway
        ui_gateway = UIGatewayActor("ui_gateway", self.actor_system, self.ui)
        self.actor_system.register(ui_gateway)
        await ui_gateway.start()

        self.callbacks.on_status_message("Session initialized.", level=StatusLevel.SUCCESS)
        self.callbacks.on_status_message(f"Using model {self.config.model}.", level=StatusLevel.SUCCESS)

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.actor_system.shutdown()

        if self._mcp_servers_cm:
            await self._mcp_servers_cm.__aexit__(exc_type, exc_val, exc_tb)

        self.callbacks.on_status_message("Session closed.", level=StatusLevel.INFO)

    async def run_chat(self, history: Optional[list[BaseMessage]] = None) -> None:
        chat_history = history or []
        try:
            await run_chat_loop(
                history=chat_history,
                model=self.config.model,
                tools=self.tools,
                instructions=self.instructions,
                callbacks=self.callbacks,
                tool_callbacks=self.tool_callbacks,
                completer=openai_complete,
                ui=self.ui,
                context_name="Orchestrator",
                actor_system=self.actor_system,
            )
        finally:
            save_orchestrator_history(self.working_directory, chat_history)

    async def run_agent(self, task: str, history: Optional[list[BaseMessage]] = None) -> Any:
        # META TOOLS: include Actor-integrated AskClient
        # Note: AskClientTool is legacy, in future everything is messages
        agent_mode_tools = [
            AskClientTool(ui=self.ui),
            *self.tools,
        ]

        desc = AgentDescription(
            name="launch_orchestrator_agent",
            model=self.config.model,
            parameters=[],  # Handled inside Task call
            tools=agent_mode_tools,
        )
        state = AgentState(history=history or [])
        ctx = AgentContext(desc=desc, state=state)

        # Trigger actor mode
        try:
            await run_agent_loop(
                ctx,
                completer=openai_complete,
                ui=self.ui,
                actor_system=self.actor_system,
                compact_conversation_at_tokens=self.config.compact_conversation_at_tokens,
            )

            # Extract output set by the OrchestratorActor
            result = ctx.state.output
            if result:
                self.callbacks.on_status_message(f"Task completed: {result.result}", level=StatusLevel.SUCCESS)
            return result
        finally:
            save_orchestrator_history(self.working_directory, ctx.state.history)
