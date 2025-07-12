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


def get_conversation_history(working_directory: Path) -> list[str]:
    conversations_file = get_conversation_history_file(working_directory)

    if not conversations_file.exists():
        logger.info("No conversations summary file found.")
        return []

    logger.info(f"Loading conversations summary from {conversations_file}.")
    conversations = json.loads(conversations_file.read_text())
    return conversations.get("summaries", [])


def save_conversation_history(working_directory: Path, summary: str):
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


def save_orchestrator_history(working_directory: Path, agent_history: list, task: str, instructions: str | None = None):
    """Save orchestrator agent history for crash recovery as a new file."""
    from datetime import datetime
    history_dir = get_orchestrator_history_dir(working_directory)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = history_dir / f"history_{timestamp}.json"
    history_data = {
        "timestamp": timestamp,
        "working_directory": str(working_directory),
        "task": task,
        "instructions": instructions,
        "agent_history": agent_history,
    }
    history_file.write_text(json.dumps(history_data, indent=2))
    logger.info(f"Saved orchestrator history for {working_directory} to {history_file}.")


def get_latest_orchestrator_history_file(working_directory: Path) -> Path | None:
    history_dir = get_orchestrator_history_dir(working_directory)
    history_files = sorted(history_dir.glob("history_*.json"), reverse=True)
    return history_files[0] if history_files else None


def load_orchestrator_history(working_directory: Path, file: str | None = None) -> dict | None:
    """Load a specific or the latest orchestrator agent history for crash recovery."""
    if file and file is not True:
        # If file is a path, use it directly; if just a filename, resolve in history dir
        file_path = Path(file)
        if not file_path.is_absolute():
            file_path = get_orchestrator_history_dir(working_directory) / file
        if not file_path.exists():
            logger.error(f"Specified history file {file_path} does not exist.")
            return None
        logger.info(f"Loading orchestrator history from {file_path}.")
        return json.loads(file_path.read_text())
    # Default: load latest
    history_file = get_latest_orchestrator_history_file(working_directory)
    if not history_file or not history_file.exists():
        logger.info("No orchestrator history file found.")
        return None
    logger.info(f"Loading orchestrator history from {history_file}.")
    return json.loads(history_file.read_text())


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
