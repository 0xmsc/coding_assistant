from pydantic import BaseModel, Field, model_validator


class MCPServerConfig(BaseModel):
    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: list[str] = Field(default_factory=list)
    prefix: str | None = None

    @model_validator(mode="after")
    def check_command_or_url(self) -> "MCPServerConfig":
        if self.command and self.url:
            raise ValueError(f"MCP server '{self.name}' cannot have both a command and a url.")
        if not self.command and not self.url:
            raise ValueError(f"MCP server '{self.name}' must have either a command or a url.")
        return self


class Config(BaseModel):
    model: str
    expert_model: str
    compact_conversation_at_tokens: int
    enable_chat_mode: bool = True
    enable_ask_user: bool = True
