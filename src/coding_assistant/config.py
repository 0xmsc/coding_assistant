from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    model: str = "gpt-4.1"
    expert_model: str = "o3"
    disable_feedback_agent: bool = False
    disable_user_feedback: bool = False
    instructions: str | None = None
    sandbox_directories: List[Path] = field(
        default_factory=lambda: [
            Path("/tmp"),
        ]
    )
    

