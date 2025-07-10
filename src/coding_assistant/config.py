from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from smolagents import Model


@dataclass
class Config:
    working_directory: Path
    model_factory: Callable[[], Model]
    expert_model_factory: Callable[[], Model]
