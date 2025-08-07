from pydantic import BaseModel, Field
from pathlib import Path
from typing import List


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: List[str]
    env: List[str] = Field(default_factory=list)


class Config(BaseModel):
    model: str
    expert_model: str
    enable_feedback_agent: bool
    enable_user_feedback: bool
    shorten_conversation_at_tokens: int
    enable_ask_user: bool
    shell_confirmation_patterns: List[str] = Field(default_factory=list)
