from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Config:
    working_directory: Path
    model: str
    expert_model: str
