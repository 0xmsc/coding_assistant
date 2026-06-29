from __future__ import annotations

import logging

from coding_assistant.infra.paths import get_log_file

logger = logging.getLogger("coding_assistant")
logger.setLevel(logging.INFO)


def setup_logging(*, console: bool = False) -> None:
    """Setup logging to the session file, optionally also to process stderr."""
    log_file = get_log_file()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
