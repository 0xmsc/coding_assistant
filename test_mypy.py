from dataclasses import dataclass
from typing import Optional, Literal

@dataclass(frozen=True, kw_only=True)
class LLMMessage:
    role: str
    content: Optional[str] = None

@dataclass(frozen=True, kw_only=True)
class SystemMessage(LLMMessage):
    content: str
    role: Literal["system"] = "system"

# This should be an error in mypy because SystemMessage.content is str, not Optional[str]
s1 = SystemMessage() 
s2 = SystemMessage(content=None) # type: ignore
s3 = SystemMessage(content="hello")
