from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from io import TextIOWrapper

from langchain_core.language_models import BaseChatModel


@dataclass
class Config:
    working_directory: Path = None
    model_factory: Callable[[], BaseChatModel] = None
    reasoning_model_factory: Callable[[], BaseChatModel] = None
    # A file into which tracing information is written (all agent output, all tool output)
    tracing_file: Optional[TextIOWrapper] = None


config: Config = Config()


def get_global_config():
    return config
