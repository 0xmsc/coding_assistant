from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    working_directory: Optional[Path] = None


config: Config = Config()


def get_global_config():
    return config
