from pydantic import BaseModel, Field
from pathlib import Path
from typing import List


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: List[str]
    env: List[str]


class Config(BaseModel):
    model: str
    expert_model: str
    enable_feedback_agent: bool
    enable_user_feedback: bool
    instructions: str | None
    readable_sandbox_directories: List[Path]
    writable_sandbox_directories: List[Path]
    mcp_servers: List[MCPServerConfig]
    shorten_conversation_at_tokens: int
    enable_ask_user: bool
    print_chunks: bool
