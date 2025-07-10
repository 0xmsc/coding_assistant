import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

from dataclasses_json import dataclass_json

logger = logging.getLogger(__name__)


@dataclass_json
@dataclass
class Config:
    model: str = "gpt-4.1"
    expert_model: str = "o3"
    disable_feedback_agent: bool = False
    disable_user_feedback: bool = False
    instructions: str | None = None
    sandbox_directories: List[Path] = field(
        default_factory=lambda: [
            "/tmp",
        ]
    )


def get_config_dir() -> Path:
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    config_dir = Path(xdg_config_home) / "coding_assistant"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _get_config_file_path() -> Path:
    return get_config_dir() / "config.json"


def _create_default_config_file_if_not_exists(config_path: Path):
    if config_path.exists():
        return

    logger.info(f"Creating default configuration file at {config_path}")

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(Config().to_json(indent=2))


def load_user_config() -> Config:
    config_path = _get_config_file_path()
    _create_default_config_file_if_not_exists(config_path)
    config = Config.from_json(config_path.read_text())

    # Convert directory strings to Path objects
    for i, directory in enumerate(config.sandbox_directories):
        if isinstance(directory, str):
            config.sandbox_directories[i] = Path(directory)

    # Expand user directories
    for i, directory in enumerate(config.sandbox_directories):
        config.sandbox_directories[i] = config.sandbox_directories[i].expanduser()

    return config
