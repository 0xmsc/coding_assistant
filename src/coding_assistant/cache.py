import logging
import os
import json
from pathlib import Path

logger = logging.getLogger("coding_assistant.cache")


def get_cache_dir() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    cache_dir = Path(xdg_cache_home) / "coding_assistant"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_conversation_history_file() -> Path:
    conversations_file = get_cache_dir() / "conversations_summary.json"
    return conversations_file


def get_conversation_history(working_directory: Path) -> list[str]:
    conversations_file = get_conversation_history_file()

    if not conversations_file.exists():
        logger.info("No conversations summary file found.")
        return []

    logger.info(f"Loading conversations summary from {conversations_file}.")
    conversations = json.loads(conversations_file.read_text())
    return conversations.get(str(working_directory), [])


def save_conversation_history(working_directory: Path, summary: str):
    conversations_file = get_conversation_history_file()

    if conversations_file.exists():
        conversations = json.loads(conversations_file.read_text())
    else:
        conversations = {}

    conversations.setdefault(str(working_directory), []).append(summary)
    conversations_file.write_text(json.dumps(conversations, indent=2))

    logger.info(f"Saved conversations summary for {working_directory} to {conversations_file}.")
