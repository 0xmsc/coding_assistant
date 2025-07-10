import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List


@dataclass
class Config:
    working_directory: Path
    model: str
    expert_model: str | None = None
    disable_feedback_agent: bool = False
    disable_user_feedback: bool = False
    instructions: str = "No instructions provided."
    sandbox_directories: List[Path] = field(default_factory=list)


logger = logging.getLogger(__name__)


def get_config_file_path() -> Path:
    """Get the path to the user configuration file."""
    return Path.home() / ".config" / "coding_assistant" / "config.json"


def create_default_config_file(config_path: Path) -> None:
    """Create a default configuration file with sensible defaults."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        "version": "1.0",
        "sandbox": {
            "additional_directories": []
        },
        "models": {},
        "features": {},
        "instructions": ""
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    logger.info(f"Created default configuration file at {config_path}")


def load_user_config(config_path: Path, base_config: Config) -> Config:
    """Load configuration from the JSON file and merge with base config."""
    user_config = {}
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            logger.info(f"Loaded user configuration from {config_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file {config_path}: {e}. Using defaults.")
            user_config = {}
        except Exception as e:
            logger.warning(f"Error loading config file {config_path}: {e}. Using defaults.")
            user_config = {}
    
    # Extract sandbox directories from user config
    sandbox_dirs = user_config.get("sandbox", {}).get("additional_directories", [])
    validated_dirs = []
    
    for dir_path in sandbox_dirs:
        try:
            path = Path(dir_path).expanduser().resolve()
            if path.exists():
                validated_dirs.append(path)
            else:
                logger.warning(f"Sandbox directory does not exist: {dir_path}")
        except Exception as e:
            logger.warning(f"Invalid sandbox directory path '{dir_path}': {e}")
    
    # Update base config with user settings
    base_config.sandbox_directories = validated_dirs
    
    # Override other settings if provided
    features = user_config.get("features", {})
    if "disable_feedback_agent" in features:
        base_config.disable_feedback_agent = features["disable_feedback_agent"]
    if "disable_user_feedback" in features:
        base_config.disable_user_feedback = features["disable_user_feedback"]
    
    if user_config.get("instructions"):
        base_config.instructions = user_config["instructions"]
    
    return base_config


# Note: merge_config_with_defaults is now deprecated in favor of the updated load_user_config
# Keeping this as a no-op for backward compatibility if needed
def merge_config_with_defaults(user_config: dict, base_config: Config) -> Config:
    """Deprecated: Use load_user_config instead."""
    logger.warning("merge_config_with_defaults is deprecated. Use load_user_config directly.")
    return base_config
