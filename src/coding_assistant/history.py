import logging
import json
from pathlib import Path
from dataclasses import asdict, is_dataclass
from coding_assistant.llm.types import message_from_dict

logger = logging.getLogger("coding_assistant.cache")


def get_project_cache_dir(working_directory: Path) -> Path:
    """Get the project-specific .coding_assistant cache directory."""
    cache_dir = working_directory / ".coding_assistant"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_orchestrator_history_file(working_directory: Path) -> Path:
    """Get the orchestrator history file for the specific project."""
    return get_project_cache_dir(working_directory) / "history.json"


def _fix_invalid_history(history: list) -> list:
    """
    Fixes an invalid history by removing trailing assistant messages with tool_calls
    that are not followed by a tool message.
    """
    if not history:
        return []

    fixed_history = list(history)
    while fixed_history:
        last_message = fixed_history[-1]
        if isinstance(last_message, dict):
            role = last_message.get("role")
            has_tool_calls = bool(last_message.get("tool_calls"))
        else:
            role = getattr(last_message, "role", None)
            has_tool_calls = bool(getattr(last_message, "tool_calls", None))

        if role == "assistant" and has_tool_calls:
            fixed_history.pop()
        else:
            break
    return fixed_history


def save_orchestrator_history(working_directory: Path, agent_history: list):
    """Save orchestrator agent history for crash recovery. Only saves agent_history."""
    history_file = get_orchestrator_history_file(working_directory)
    fixed_history = _fix_invalid_history(agent_history)

    serializable_history = []
    # Argument 1 to "asdict" has incompatible type "DataclassInstance | type[DataclassInstance]"; expected "DataclassInstance"
    for msg in fixed_history:
        if isinstance(msg, dict):
            serializable_history.append(msg)
        elif is_dataclass(msg) and not isinstance(msg, type):
            serializable_history.append(asdict(msg))
        else:
            serializable_history.append(msg)

    history_file.write_text(json.dumps(serializable_history, indent=2))

    logger.info(f"Saved orchestrator history for {working_directory} to {history_file}.")


def get_latest_orchestrator_history_file(working_directory: Path) -> Path | None:
    history_file = get_orchestrator_history_file(working_directory)
    return history_file if history_file.exists() else None


def load_orchestrator_history(file: str | Path) -> list | None:
    """Load orchestrator agent history from a specific file. Returns agent_history list or None."""
    file_path = Path(file)
    if not file_path.exists():
        logger.error(f"Specified history file {file_path} does not exist.")
        return None
    logger.info(f"Loading orchestrator history from {file_path}.")
    data = json.loads(file_path.read_text())
    return [message_from_dict(m) for m in data]
