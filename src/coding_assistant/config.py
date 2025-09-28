from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str]
    env: list[str] = Field(default_factory=list)


class Config(BaseModel):
    model: str
    expert_model: str
    enable_user_feedback: bool
    shorten_conversation_at_tokens: int
    enable_ask_user: bool
