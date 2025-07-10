from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable

from langchain_core.language_models import BaseChatModel


@dataclass
class Config:
    working_directory: Path = None
    model_factory: Callable[[], BaseChatModel] = None
    expert_model_factory: Callable[[], BaseChatModel] = None


config: Config = Config()


def get_global_config():
    return config
