from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Config:
    working_directory: Path
    model: str
    expert_model: str | None = None
    disable_feedback_agent: bool = False
    disable_user_feedback: bool = False
    instructions: str = "No instructions provided."
