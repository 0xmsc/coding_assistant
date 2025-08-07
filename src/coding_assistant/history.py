import logging
import os
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("coding_assistant.cache")


def get_project_cache_dir(working_directory: Path) -> Path:
    """Get the project-specific .coding_assistant cache directory."""
    cache_dir = working_directory / ".coding_assistant"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_conversation_history_file(working_directory: Path) -> Path:
    """Get the conversation history file path for the specific project."""
    conversations_file = get_project_cache_dir(working_directory) / "conversations_summary.json"
    return conversations_file


def get_conversation_summaries(working_directory: Path) -> list[str]:
    conversations_file = get_conversation_history_file(working_directory)

    if not conversations_file.exists():
        logger.info("No conversations summary file found.")
        return []

    logger.info(f"Loading conversations summary from {conversations_file}.")
    conversations = json.loads(conversations_file.read_text())
    return conversations.get("summaries", [])


def save_conversation_summary(working_directory: Path, summary: str):
    conversations_file = get_conversation_history_file(working_directory)

    if conversations_file.exists():
        conversations = json.loads(conversations_file.read_text())
    else:
        conversations = {"summaries": []}

    conversations["summaries"].append(summary)
    conversations_file.write_text(json.dumps(conversations, indent=2))

    logger.info(f"Saved conversations summary for {working_directory} to {conversations_file}.")


def get_orchestrator_history_dir(working_directory: Path) -> Path:
    """Get the orchestrator history directory for the specific project."""
    history_dir = get_project_cache_dir(working_directory) / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir


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
        if last_message["role"] == "assistant" and "tool_calls" in last_message:
            fixed_history.pop()
        else:
            break
    return fixed_history


def save_orchestrator_history(working_directory: Path, agent_history: list):
    """Save orchestrator agent history for crash recovery as a new file. Only saves agent_history."""
    history_dir = get_orchestrator_history_dir(working_directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = history_dir / f"history_{timestamp}.json"
    fixed_history = _fix_invalid_history(agent_history)
    history_file.write_text(json.dumps(fixed_history, indent=2))

    logger.info(f"Saved orchestrator history for {working_directory} to {history_file}.")


def get_latest_orchestrator_history_file(working_directory: Path) -> Path | None:
    history_dir = get_orchestrator_history_dir(working_directory)
    history_files = sorted(history_dir.glob("history_*.json"), reverse=True)
    return history_files[0] if history_files else None


def load_orchestrator_history(file: str | Path) -> list | None:
    """Load orchestrator agent history from a specific file. Returns agent_history list or None."""
    file_path = Path(file)
    if not file_path.exists():
        logger.error(f"Specified history file {file_path} does not exist.")
        return None
    logger.info(f"Loading orchestrator history from {file_path}.")
    return json.loads(file_path.read_text())


def clear_orchestrator_history(working_directory: Path):
    """Clear all orchestrator agent history files after successful completion."""
    history_dir = get_orchestrator_history_dir(working_directory)
    for file in history_dir.glob("history_*.json"):
        file.unlink()
    logger.info(f"Cleared all orchestrator history for {working_directory}.")


def trim_orchestrator_history(working_directory: Path, keep: int = 10):
    """Keep only the latest N orchestrator history files."""
    history_dir = get_orchestrator_history_dir(working_directory)
    history_files = sorted(history_dir.glob("history_*.json"), reverse=True)
    for file in history_files[keep:]:
        file.unlink()
    logger.info(f"Trimmed orchestrator history for {working_directory}, kept {keep} most recent files.")
