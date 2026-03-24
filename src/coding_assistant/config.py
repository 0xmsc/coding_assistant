from pydantic import BaseModel, Field, model_validator


class MCPServerConfig(BaseModel):
    """Configuration for connecting to one MCP server."""

    name: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: list[str] = Field(default_factory=list)
    prefix: str | None = None

    @model_validator(mode="after")
    def check_command_or_url(self) -> "MCPServerConfig":
        """Require exactly one connection backend: command or URL."""
        if self.command and self.url:
            raise ValueError(f"MCP server '{self.name}' cannot have both a command and a url.")
        if not self.command and not self.url:
            raise ValueError(f"MCP server '{self.name}' must have either a command or a url.")
        return self
