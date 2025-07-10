from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Any # Placeholder, adjust if Model type is known
Model = Any # Placeholder


@dataclass
class Config:
    working_directory: Path
    model_factory: Callable[[], Model]
    expert_model_factory: Callable[[], Model]
