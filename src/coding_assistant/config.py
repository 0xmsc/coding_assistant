from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str]
    env: list[str] = Field(default_factory=list)


class Config(BaseModel):
    model: str
    expert_model: str
    shorten_conversation_at_tokens: int
    enable_chat_mode: bool = True
