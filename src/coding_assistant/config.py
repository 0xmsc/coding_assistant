from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: List[str] = field(default_factory=list)  # List of env var keys to pass through


@dataclass
class Config:
    model: str
    expert_model: str
    enable_feedback_agent: bool
    enable_user_feedback: bool
    instructions: str | None
    sandbox_directories: List[Path]
    mcp_servers: List[MCPServerConfig]
    shorten_conversation_at_tokens: int
    enable_ask_user: bool


